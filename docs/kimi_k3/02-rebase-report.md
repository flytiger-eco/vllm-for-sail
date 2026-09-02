# Kimi K3 上游 PR rebase 报告

> 目标分支：`feat/kimi-k3-ppu`（fork 自 `rebase/v0.26.0` @ `d4824f299`）
> 来源：`vllm-project/vllm` PR #50000 分支 `kimi-k3`，共 3 个提交
> 结果：**3 个提交全部落地，11 个冲突文件全部解决，263 files / +40,125 / −2,414**

---

## 1. 起始状态与冲突根因

```
upstream main
   dcfebf93f  ←── 我们分支的上游锚点（merge-base）
       │
       ├── 195 commits (上游漂移) ──→ 0ba2aa35a  ←── PR #50000 的 base
       │                                  │
       │                                  ├── f68f4fdde  kimi-k3        (主提交)
       │                                  ├── a43ab53c8  cargo.toml
       │                                  └── 31776d0c9  update dependency
       │
       └── 54 commits (我们的 PPU 提交) ──→ d4824f299  = feat/kimi-k3-ppu 起点
```

**冲突有两个独立来源，必须区分对待**：

| 来源 | 数量 | 处理原则 |
|---|---|---|
| **A. 上游漂移**：PR 的 pre-image 是 195 个上游提交之后的代码，我们的分支是之前的 | 7 处 | 逐个判断：PR 依赖的新上游能力我们是否具备。具备→取 PR 侧；不具备→取我方侧并记录能力缺口 |
| **B. 真实语义碰撞**：PPU 定制代码与 PR 改动同一段逻辑 | 2 处（`mxfp4.py`）| 手工合并，PPU 分支优先、PR 分支补齐 |

操作方式：`git cherry-pick -x` 三个提交（而非 `git rebase`），保留 `(cherry picked from ...)` 溯源信息。

---

## 2. 冲突清单与解决方式

### 2.1 `vllm/model_executor/layers/quantization/mxfp4.py` — 冲突 ①（真实语义碰撞，最重要）

**位置**：`Mxfp4Config.get_quant_method()` 的 `RoutedExperts` 分支。

**双方**

| 侧 | 内容 |
|---|---|
| 我方 (PPU) | ① `ignored_layers` 命中 → `UnquantizedFusedMoEMethod`（MTP block 专家是 BF16 未量化）；② `is_ppu() and not isinstance(self, GptOssMxfp4Config)` → `Mxfp4MoEMethod`；③ 否则 `GptOssMxfp4MoEMethod` |
| PR | 抽象成 `return self._make_moe_method(layer.moe_config)`；基类 `Mxfp4Config._make_moe_method → Mxfp4MoEMethod`，`GptOssMxfp4Config` 覆写 → `GptOssMxfp4MoEMethod` |

**解决**：保留我方 ①，把 ②③ 换成 PR 的 `self._make_moe_method(...)`。

**理由**：PR 的多态派发**完全覆盖**了我们 ② 的意图 —— 我们那句 `is_ppu() and not GptOss` 本质就是"非 GPT-OSS 的 mxfp4 走 `Mxfp4MoEMethod`"，而 PR 把它提升为所有平台的默认行为并用子类覆写处理 GPT-OSS。继续保留 `is_ppu()` 判断会变成死代码。① 必须留：`ignored_layers → Unquantized` 是 PPU 混精 checkpoint 的专属需求（MTP 块专家按 BF16 形状加载），上游没有，PR 也没有。

**副作用（需知晓）**：在 CUDA 上，非 GPT-OSS 的 mxfp4 checkpoint 从此走 `Mxfp4MoEMethod` 而不再是 `GptOssMxfp4MoEMethod`。这是 PR 的既定意图（K3 需要动态读取 SwiGLU 参数），不是我们引入的行为变化。

---

### 2.2 `vllm/model_executor/layers/quantization/mxfp4.py` — 冲突 ②（真实语义碰撞）

**位置**：`Mxfp4MoEMethod.__init__()` 的后端选择。

**双方**

| 侧 | 内容 |
|---|---|
| 我方 (PPU) | `if is_ppu()`：非 sm80 → `select_mxfp4_moe_backend(moe, activation_key=kMxfp4Dynamic)`（W4A4）；sm80 (810E) → `select_mxfp4_moe_backend(moe)`（W4A16 + Marlin）。否则 → `select_deepseek_v4_mxfp4_moe_backend` |
| PR | 新增 `self.is_k3_situ_aiter`（ROCm gfx950 + SiTU + AITER）→ `AITER_MXFP4_BF16`；否则 `select_deepseek_v4_mxfp4_moe_backend` |

**解决**：三路 if/elif/else，PPU 分支置首，其后是 PR 的 ROCm 分支，最后是默认分支；同时保留 PR 新增的
`self.is_k3_situ_aiter = _use_k3_situ_aiter(moe)` 无条件赋值和 `self.experts_cls: type[mk.FusedMoEExperts] | None` 类型标注。

```python
self.is_k3_situ_aiter = _use_k3_situ_aiter(moe)
self.experts_cls: type[mk.FusedMoEExperts] | None
if current_platform.is_ppu():
    ...                                    # 我方：W4A4 / W4A16 按 arch 分流
elif self.is_k3_situ_aiter:
    ...                                    # PR：ROCm gfx950 AITER
else:
    ...                                    # 上游默认
```

**理由**：两个分支互斥（`_use_k3_situ_aiter` 首行就是 `if not current_platform.is_rocm(): return False`），所以顺序上无冲突；PPU 前置是为了**逐字保持我方现有行为**。`is_k3_situ_aiter` 必须在分支外无条件赋值 —— 该文件后续 4 处（权重形状、`_setup_kernel_k3_situ`、prepare/finalize）都读它，若只在 else 分支赋值，PPU 路径会 `AttributeError`。

---

### 2.3 `vllm/model_executor/warmup/qwen_triton_warmup.py`（上游漂移）

**双方**：我方持有 zero-KV-block warmup 的**旧版**实现（`_ZeroKvWarmupConfig(page_size_el, block_size, n_segs)`，4 元组 `_meta`）；PR 把这整块**删除**（其 pre-image 是上游新版 `seg_page_sizes/max_chunks`，5 元组 `_meta`）。

**解决**：取 PR 侧（删除），结果与 PR 版本逐字节一致。

**理由**：PR 把这段能力搬到了 `kernel_warmup.py` 里的 `zeroer.warmup(num_blocks)`，而
`KVBlockZeroer.warmup()` 正是 PR 在 `vllm/v1/worker/utils.py` 里新增的（+6 行，已干净应用到我们分支，
现位于 `vllm/v1/worker/utils.py:241`）。所以删除**不丢功能**，只是换了归属。保留我方旧实现反而会与
新的调用点重复预热。另外确认：我方在该文件的唯一 delta 就是这块旧版实现，没有 PPU 定制。

---

### 2.4 `vllm/model_executor/warmup/kernel_warmup.py`（上游漂移）

**双方**：PR 侧在文件末尾加

```python
if worker.vllm_config.kernel_config.enable_jit_warmup:
    kimi_k3_triton_warmup(worker)
    fa4_cutedsl_warmup(worker)
    sparse_mla_triton_warmup(worker)
```

我方侧为空。

**解决**：只保留 `kimi_k3_triton_warmup(worker)`，无条件调用。

**理由**：`kernel_config.enable_jit_warmup`、`fa4_cutedsl_warmup`、以及新命名的
`sparse_mla_triton_warmup` 都是我们分支尚不存在的上游能力（我们仍是
`sparse_mla_triton_warmup_if_needed(worker)`，且已在第 108 行无条件调用）。照搬会立刻
`AttributeError`。K3 的预热本身必须保留，且它自带双重短路：`if not current_platform.is_cuda(): return`
以及 `if _get_kda_layer(worker) is None: return`，所以无条件调用对非 K3 模型零开销。

> 注：`current_platform.is_cuda()` 在本仓库对 PPU 返回 **True**（`interface.py:193`
> `_enum == CUDA or _enum == PPU`），所以 PPU 上这条预热**会执行**。见交接报告。

---

### 2.5 `vllm/v1/kv_cache_interface.py`（上游漂移，可并集）

**双方**：同一个 `MLAAttentionSpec` dataclass，我方加了 `indexer_n_head` / `indexer_q_head_dim`（DSV4 sparse MLA indexer），PR 加了 `non_causal_multi_token_decode`。

**解决**：**并集**，三个字段全留。

**理由**：字段互不相干，都是带默认值的 kw-only 字段，无顺序约束。`non_causal_multi_token_decode`
是 K3 的 DSpark/MTP 草稿组共用 MLA 层的必要开关（`mla_attention.py` 与
`v1/attention/backends/mla/prefill/base.py` 都读它）；`indexer_*` 是我们 DSV4 路径在用的。

---

### 2.6 `vllm/model_executor/layers/attention/mla_attention.py`（上游漂移）

**双方**：PR 在 `__init__` 里加 `self.non_causal_multi_token_decode = non_causal_multi_token_decode`；我方该位置是空行。

**解决**：取 PR 侧。

**理由**：形参 `non_causal_multi_token_decode: bool = False` 已随 PR 干净应用到签名里
（`mla_attention.py:378`），只是赋值语句落在了冲突区。不赋值会导致
`MLAAttentionSpec` 构造时读不到属性。纯粹是上下文对齐问题，无语义抉择。

---

### 2.7 `vllm/model_executor/layers/fused_moe/modular_kernel.py`（上游漂移，注释冲突）

**双方**：同一段注释的两种措辞。我方："Gated by VLLM_MOE_SKIP_PADDING (off by default) because ... not all MoE backends support yet."；PR："This requires the experts kernel to treat topk_id == -1 as a skip sentinel."

**解决**：保留我方注释。

**理由**：下方代码在两侧**完全一致**（`if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():`），
即 env 开关在 PR 里也仍然存在。我方注释显式说明了"默认关闭"这一事实，信息量更大且准确。零功能影响。

---

### 2.8 `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`（上游漂移，**必须取我方**）

**双方**：PR 侧文件尾部有

```python
if current_platform.is_cpu():
    from ...ops.cpu.causal_conv1d import causal_conv1d_fn_cpu, causal_conv1d_update_cpu
    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
```

我方侧无此块。

**解决**：取我方侧（不引入该块）。

**理由**：这是上游后来加的 CPU 回退，PR 只是把它当上下文带过来。我们分支的
`vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py` 只导出
`causal_conv1d_torch` / `causal_conv1d_update_torch`，**没有 `*_cpu` 这两个名字** ——
照搬会在 CPU 平台上直接 `ImportError`。

PR 对该文件的实质改动（`do_not_specialize_on_alignment=["num_cache_lines"]`、
`launch_pdl` 参数与 `gdc_wait/gdc_launch_dependents`、`num_cache_lines` 去 constexpr）**已全部干净应用**，
这是 K3 KDA prefill/decode 卷积所需的，未受影响。

---

### 2.9 `vllm/v1/structured_output/__init__.py`（PR 自身改动 + 上游漂移叠加）

**双方**

| 侧 | `should_advance()` 中 "reasoning 刚结束" 的处理 |
|---|---|
| 我方（旧上游） | 只对 `STRUCTURAL_TAG` 且开了投机解码时同步推进 FSM 并 `return True`，其他类型延迟到下一轮 |
| PR | 一律记录 `reasoning_end_token_index` 并 `return True`；`_find_reasoning_end_index` 从"线性扫 `is_reasoning_end_streaming`"改为"以 prompt 末尾为 floor 对 `is_reasoning_end` 做**二分查找**" |

**解决**：取 PR 侧。

**理由**：这是 PR 的**主动修复**（不是纯漂移）：PR 的注释解释了延迟推进为什么会让 FSM 与已发出的
token 脱同步（token 是在 grammar 初态下采样的但从未被 accept，下一轮从初态重新推导会重复
grammar 前缀，如 `{"` + `{"city": ...}`）。K3 自带 reasoning parser + structural tag registry
（`structural_tag_registry.py` +263），正是这条路径的重度使用者。同时
`_find_reasoning_end_index` 的签名从 `start` 改成 `floor`，调用点也在 PR 侧，取我方会签名不匹配。

已核对：`reasoner.is_reasoning_end(...)`（二分所需）在我们分支的 `ReasoningParser` 上存在；
本文件我方唯一的本地改动是 `e5949f100 [Bugfix] handle grammar compilation failures`（grammar 编译失败处理），
与本冲突区不重叠，已完整保留。

---

### 2.10 `tests/v1/structured_output/test_reasoning_structured_output.py`（上游漂移，**必须取我方**）

**双方**：PR 侧带来约 120 行测试（`test_should_advance_uses_new_token_ids_when_provided`、
`test_should_advance_without_new_token_ids_falls_back`、`test_should_advance_trims_reasoning_prefix_for_json`）；我方侧为空。

**解决**：取我方侧（不引入这些测试），结果与 rebase 前逐字节一致。

**理由**：这些测试是**上游的**（我方该文件与 `dcfebf93f` 完全相同 → 纯漂移），它们调用
`manager.should_advance(request, new_token_ids=...)`。而 `should_advance` 的 `new_token_ids`
形参是我们分支尚未具备的上游能力（PR 自身没有引入它，所以那个 hunk 不冲突、也不会被应用）。
引入这些测试只会得到一批必然失败的用例。

**代价（已知）**：PR 对本文件的实质改动只有 3 行 —— 给 `MarkerReasoner` 加
`is_reasoning_end` 方法，而宿主测试 `test_should_advance_trims_reasoning_prefix_for_json`
在我们分支不存在，因此这 3 行事实上落空。这不影响生产代码，只是我们暂时缺少对
§2.9 新二分逻辑的单测覆盖。**后续补齐建议**：等我们跟进上游 `should_advance(new_token_ids=)`
之后再一并把这三个测试拉过来。

---

### 2.11 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py`（PR 自身改动）

**双方**

| 侧 | 分片 xfer handle 的 key |
|---|---|
| 我方 | `tp_ratio not in self.src_xfer_handles_by_tp_ratio`，且 `not self.use_mla` 才走多源读 |
| PR | `split_key = (tp_ratio, remote_block_size)` 作 key，条件放宽为 `not self.use_mla or len(plan.all_source_ranks) > 1` |

**解决**：取 PR 侧。

**理由**：**强制性一致性** —— PR 的其余改动已干净应用，同一函数下游第 1668 行已经在用
`self.src_xfer_handles_by_tp_ratio[split_key].append(handle)`，字典的类型标注也已改成
`dict[tuple[int, int], list[int]]`（第 465 行），`plan.all_source_ranks` 也已存在（第 207 行）。
保留我方的 `tp_ratio` 作 key 会与写入端不匹配，直接 `KeyError`。

语义上这正是 K3 需要的 HMA（hybrid memory allocation）路径：MLA 部分是复制的、读一次即可，
而 SSM/mamba state 是按远端 TP rank 分片的，必须每个源 rank 读一次。

---

### 2.12 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py`（PR 自身改动 + **一处主动补齐**）

**冲突本体**

| 侧 | 内容 |
|---|---|
| 我方 | `if tp_ratio < 0 and not self.use_mla:` + `assert remote_block_size == self.block_size` |
| PR | `if tp_ratio < 0 and (not self.use_mla or len(read_specs) > 1):`（去掉 assert） |

**解决**：取 PR 侧。理由同 §2.11 —— 紧随其后的代码已在用 `split_key = (tp_ratio, remote_block_size)`，
且去掉 assert 正是为了支持"远端 block_size ≠ 本地 block_size"。

**额外的主动修改（不是冲突，但必须做）**：同函数上方有一处未冲突的旧代码

```python
- if self.use_mla and tp_ratio < 0:
+ if self.use_mla and tp_ratio < 0 and not self._has_mamba:
      assert len(read_specs) == 1
```

**为什么改**：这行在 PR 的 pre-image 里已经是 `and not self._has_mamba`（上游漂移，PR 未触碰
→ 不冲突 → git 保留了我方旧版）。但它与刚取入的 `len(read_specs) > 1` 分支**直接矛盾**：
对 K3 这种 hybrid MLA+SSM 模型，`read_specs` 必然 > 1，旧 assert 会先炸掉。
`self._has_mamba` 由 PR 的 `base_worker.py` 改动引入，已存在于我们分支（`base_worker.py:311` 等 10 处）。
不改这一行，整个 PR 的 HMA 支持在 P/D 分离场景下就是死的。

---

## 3. 干净应用但值得记录的高风险区

以下文件没有冲突（git 自动合并成功），但改动幅度大、且对 PPU 有影响，需要在设备上验证：

| 文件 | 改动 | 关注点 |
|---|---|---|
| `vllm/distributed/device_communicators/custom_all_reduce.py` (+301) | MNNVL multicast + Lamport buffer 初始化 | PPU 没有 MNNVL；`_init_mnnvl_buffer` 是否会在 PPU 上抛异常需验证。`custom_all_gather`/`custom_reduce_scatter` 在 `cuda_communicator.py` 里返回 `None` 即回退 NCCL |
| `vllm/config/parallel.py` | **删除**了 `nnodes > 1 → disable_custom_all_reduce` | 多节点 PPU 部署现在会尝试 custom AR；若 PPU 的 IPC 探测不严格，可能踩坑 |
| `vllm/config/vllm.py` (+50) | `default_v2_model_runner_architectures()`，ROCm 排除 K3 的 V2 runner | PPU 未加入排除名单，K3 会默认走 V2 model runner |
| `vllm/v1/core/kv_cache_utils.py` (+24) | 同类型层按 `spec.merge()` 可行性进一步合并分组 | 影响所有 hybrid 模型的 KV group 划分，含 PPU DSV4 |
| `vllm/v1/core/sched/scheduler.py` | mamba align 模式允许 sub-block 推进；**删除**了 `block_size <= max_num_batched_tokens` 断言 | 我们 PPU 上 mamba-align 的调度行为会变 |
| `vllm/platforms/interface.py` (+6) | MLA 模型的 `kernel_block_alignment_size` 抬到 ≥128 | PPU 继承 `NvmlCudaPlatform`，会生效；需确认 PPU FlashMLA 的 block 约束兼容 |
| `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` (±771) | Kimi-Linear GDN 层大重构 | 我们若有 Kimi-Linear 相关 PPU 分支需重新对齐 |
| `vllm/model_executor/models/kimi_k25_vit.py` (+229) | 3D 位置嵌入、packed patch merge、encoder CUDA graph | K2.5 ViT 在 PPU 上的现状需回归 |
| `csrc/` 全部新 kernel | `custom_all_gather_reduce_scatter*`、`fused_kimi_k3_mla_*`、`kimi_k3/{attn_res,fused_kda_decode}` | 编译门控见下 |

### 编译门控现状（对我们有利）

```
FUSED_KDA_DECODE_ARCHS   ← CUDA≥13.0 ∩ {9.0a, 10.0f, 12.0f}   → 宏 VLLM_ENABLE_FUSED_KDA_DECODE
KIMI_K3_ATTN_RES_ARCHS   ← CUDA≥13.0 ∩ {10.0f}                → 宏 VLLM_ENABLE_KIMI_K3_ATTN_RES
FLASH_KDA_ARCHS          ← 独立扩展 _flashkda_C，不匹配则 add_custom_target 空目标
```

三者都是"arch 不匹配就整段不编译"，且 Python 侧都有 Triton 回退 + `hasattr(torch.ops._C, ...)` 运行时探测。
**所以 PPU 编译不会因为这些新 kernel 失败**，只会走 Triton 路径。

---

## 4. 验证结果

| 检查 | 结果 |
|---|---|
| 冲突标记残留（全仓 `*.py/*.cu/*.cuh/*.h/*.cpp/*.rs/*.toml/*.cmake/*.yaml/*.txt`） | 0 |
| 改动的 201 个 Python 文件 AST 解析 | 全部通过（用 py3.9 解析，等于更严的语法约束） |
| 跨模块 `from vllm.X import Y` 静态解析（160 个 vllm 文件） | 无真实缺失；报告项全部是 `vllm.platforms` / `vllm.distributed` 的 lazy `__getattr__` 与 py3.9 无法解析 `match` 语句导致的假阳性 |
| 关键依赖能力抽查 | `kernel_config.moe_backend`（含 `"deep_gemm_mega_moe"`）✓、`MoEActivation.SITU` ✓、3 个新 env ✓、`CuTeDSLCompileUnit`/`register_cutedsl_warmup_provider` ✓、`get_num_attention_heads_from_layers` ✓、`FusedMoE(routed_output_transform/runner_cls/runner_args/activation_situ_*/is_sequence_parallel)` ✓、`MoERunner._shared_experts/_fused_output_is_reduced/apply_routed_output_transform/_unpack` ✓、`FusedMoEConfig.activation_situ_*` ✓、`KVBlockZeroer.warmup` ✓ |
| 旧模块引用残留（`vllm/model_executor/models/kimi_linear.py` 已删除） | 0 处残留引用 |
| `fused_q_kv_rmsnorm` 迁移（`deepseek_v4/common/ops/` → `models/common/ops/`） | 旧路径已删、新路径已建、调用点已更新，0 处残留 |
| E501（>88 字符且无 `# noqa`）在 11 个解决文件中的增量 | **0 新增**（`mxfp4.py` 反而 −1） |

未执行（本机无 `.venv`、无 GPU、无 ruff/pre-commit）：`pre-commit run --all-files`、
`pytest`、编译。这些必须在有环境的机器上跑，建议顺序见 §5。

---

## 5. 下一步建议（按顺序）

1. **在有环境的机器上补 lint**
   ```bash
   uv venv --python 3.12 && source .venv/bin/activate
   uv pip install -r requirements/lint.txt
   pre-commit run --all-files          # 注意 .pre-commit-config.yaml 已把 kimi_k3/*/ops/third_party/ 排除
   pre-commit run mypy-3.12 --all-files --hook-stage manual
   ```
2. **纯 Python 导入冒烟**（不需要 GPU）：
   ```bash
   python -c "import vllm.transformers_utils.configs.kimi_k3"
   python -c "import vllm.reasoning.kimi_k3_reasoning_parser, vllm.tool_parsers.kimi_k3_tool_parser, vllm.renderers.kimi_k3"
   ```
3. **PPU 编译**：确认 `VLLM_ENABLE_FUSED_KDA_DECODE` / `VLLM_ENABLE_KIMI_K3_ATTN_RES` 两个宏在 PPU 工具链下**未定义**，
   `_flashkda_C` 落到空 target，`situ_and_mul` / `masked_situ_and_mul` / `concat_and_cache_mla_grouped`
   这三个新算子编进 `_C_stable_libtorch`。
4. **回归我们已有的 PPU 模型**（DSV4 / GPT-OSS MXFP4），因为 §2.1–2.2 改了 MXFP4 派发、
   §3 改了 KV group 合并与调度器 align 逻辑。
5. **K3 适配开工**：见 `03-ppu-handoff.md`。
