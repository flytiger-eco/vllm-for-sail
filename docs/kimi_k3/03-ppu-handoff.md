# Kimi K3 PPU 适配交接报告

> 前置阅读：`01-upstream-pr-50000-analysis.md`（上游做了什么）、`02-rebase-report.md`（我们的分支现在是什么状态）
> 分支：`feat/kimi-k3-ppu`
> 命题：**基于 PPU FlashMLA + PPU DeepGemm MoE + KDA，我能复用多少 CUDA/AMD 的 cu kernel 与 triton kernel，还差什么？**

---

## 0. 结论先行

| 三条主线 | 能否复用现有 PPU 资产 | 还差什么（一句话） |
|---|---|---|
| **MLA** | ✅ 高度复用。K3 的 MLA decode 走的是**通用** `MLAAttentionImpl.forward_mqa(q, kv_cache, attn_metadata, layer) -> (out, lse)` 契约，`FLASHMLA` / `TRITON_MLA` 都实现了它，且两者都在 PPU 的 MLA backend 优先级表里 | 6 个 `fused_kimi_k3_mla_*` 融合 epilogue 算子需要在 PPU 上有等价实现或 Triton/eager 回退（**PR 没给 Triton 回退，这是最大的缺口**） |
| **MoE** | ⚠️ 中度复用。`FusedMoE + LatentMoERunner` 是纯 Python 编排，直接可用；专家 kernel 走 `PPUDeepGemmExpertsMXFP4` | ① 该类**不支持 `MoEActivation.SITU`**（白名单里没有）；② 只支持 W4A4（`kMxfp4Static, kMxfp4Dynamic`），而 K3 是 **W4A16 weight-only**；③ `situ_and_mul` CUDA 算子需在 PPU 工具链编出来；④ MegaMoE（`fp8_fp4_mega_moe`）PPU 侧不存在，必须走非 MegaMoE 路径 |
| **KDA** | ✅ 最高度复用。prefill / spec-decode / 回退 decode **全部是 Triton**，且上游已提供 NVIDIA 与 AMD 两套可对照的移植版 | fused decode CUDA kernel 与 FlashKDA 都用不了（arch 门控自动关闭），先跑 Triton 路径即可；`gather_initial_states`、`FusedRMSNormGated`、`causal_conv1d` 的 PDL 分支需验证 |
| **AttnRes** | ✅ 完全复用 Triton | 原生 CUDA 版仅 SM100，PPU 自动落到 Triton；`amd/ops/attn_res.py` 是更简洁的参考实现 |

**一句话**：**KDA 和 AttnRes 几乎零成本复用（纯 Triton），MoE 需要在 `PPUDeepGemmExpertsMXFP4` 上开 SiTU + 决定 W4A16 策略，MLA 的真正工作量集中在那 6 个 fused KV-cache epilogue 算子。**

---

## 1. 现有 PPU 资产盘点（可直接站在上面）

### 1.1 平台层

`vllm/platforms/ppu.py`（164 行，`PPUPlatform(NvmlCudaPlatform)`）：

```python
_enum = PlatformEnum.PPU;  device_type = "cuda";  dispatch_key = "CUDA";  dist_backend = "nccl"
MLA backend 优先级 = [FLASHMLA, TRITON_MLA, FLASHMLA_SPARSE]
非 MLA 优先级     = [FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION]
ViT backend       = [FLASH_ATTN, TRITON_ATTN, TORCH_SDPA]   (sm80+)
use_custom_allreduce() -> False
apply_config_platform_defaults(): flash_attn_max_num_splits_for_cuda_graph=0,
                                  custom_ops += ["+sparse_attn_indexer", "+quant_fp8"]
```

**三个关键推论**

1. `current_platform.is_cuda()` 对 PPU **返回 True**（`interface.py:193`：`_enum == CUDA or _enum == PPU`）。
   所以所有 `if not current_platform.is_cuda(): return` 的守卫在 PPU 上**不会**短路 ——
   例如 `kimi_k3_triton_warmup()` 会在 PPU 上真正执行。这是好事（预热有效），
   但也意味着任何"CUDA-only"的假设会直接落到 PPU 上。
2. `is_cuda_alike()` 也包含 PPU，所以 `SituAndMul.forward_cuda` 会走 `torch.ops._C.situ_and_mul`。
3. `use_custom_allreduce() -> False` ⇒ **整套 custom AR / AG / RS / MNNVL Lamport 在 PPU 上天然禁用**。
   `sequence_parallel.py` 的 `_custom_collective()` 拿不到 `ca_comm` 会返回 `None`，
   自动回退到 `tensor_model_parallel_all_gather / reduce_scatter`（NCCL）。
   **K3 的 sequence parallel 在 PPU 上开箱可用（走 NCCL），无需移植任何 collective kernel。**

### 1.2 MLA 资产

| 文件 | 内容 |
|---|---|
| `vllm/models/deepseek_v4/ppu/flashmla.py` (15.7K) | `DeepseekV4FlashMLAAttention(DeepseekV4Attention)`：`forward_mqa` / `_forward_decode` / `_forward_prefill` / `get_padded_num_q_heads`（PPU 支持任意 head 数，**无需 pad 到 64/128**）/ `_o_proj` |
| `vllm/models/deepseek_v4/ppu/ops/o_proj.py` | `deep_gemm_fp8_o_proj`，融合 inv-RoPE + fp8/int8 量化 + `fp8_einsum`；含 SM90/SM100 的 scale layout 分流 |
| `vllm/v1/attention/ops/ppu_mla_sparse.py` | `ppu_sparse_attn_indexer`（+ fake），DSV4 sparse MLA |
| `vllm/v1/attention/backends/mla/flashmla.py` | 通用 `FlashMLAImpl.forward_mqa`（PPU 走这条） |
| `vllm/v1/attention/backends/mla/triton_mla.py` | 通用 `TritonMLAImpl.forward_mqa`，`supports_quant_query_input = False` |
| `vllm/vllm_flash_attn/ppu/` | PPU 自己的 FA interface + `ops/triton/rotary.py` |

### 1.3 MoE 资产

| 文件 | 内容 |
|---|---|
| `vllm/model_executor/layers/fused_moe/experts/ppu_deep_gemm_moe.py` | `PPUDeepGemmExperts`（fp8/int8）、`PPUDeepGemmExpertsMXFP4`（mxfp4） |
| `vllm/model_executor/layers/fused_moe/experts/ppu_batched_deep_gemm_moe.py` | batched（EP a2a）变体 |
| `vllm/utils/ppu_deep_gemm.py` | PPU DeepGEMM 绑定：`m_grouped_fp4_gemm_nt_nopad`、`fp4_m_grouped_gemm_nt_masked`、`bf16_m_grouped_gemm_nt_masked`、`m_grouped_bf16_gemm_nt_nopad`、`transform_sf_into_required_layout`、`fp8_einsum`/`int8_einsum`、`fp8_paged_mqa_logits` 等 |
| `vllm/model_executor/layers/quantization/utils/ppu_mxfp4_utils.py` | PPU mxfp4 权重/scale 处理 |
| `vllm/model_executor/warmup/ppu_deep_gemm_warmup.py` | DeepGEMM 预热 |

**注意**：`ppu_deep_gemm.py` 里**没有** `fp8_fp4_mega_moe` / `transform_weights_for_mega_moe` /
`get_symm_buffer_for_mega_moe`。⇒ `KimiK3MegaMoEExperts` 路径在 PPU 上不可用，
必须保证 `kernel_config.moe_backend != "deep_gemm_mega_moe"`。

### 1.4 平台隔离模式（照抄即可）

`vllm/models/deepseek_v4/__init__.py` 已经建立了标准范式：

```python
if current_platform.is_rocm():   from .amd.model import ...
elif current_platform.is_xpu():  from .xpu.model import ...
elif current_platform.is_ppu():  from .ppu.model import ...   # ← 我们要加的分支
else:                            from .nvidia.model import ...
```

而 `vllm/models/deepseek_v4/ppu/model.py` 只有 73 行 —— 它**继承** NVIDIA 实现，只覆写
`packed_modules_mapping` 和权重 mapper 的选择。这就是我们对 K3 应该采取的形态。

---

## 2. 逐 kernel 复用判定矩阵

### 2.1 CUDA `.cu` 算子

| 算子 | 源 | PPU 判定 | 说明 |
|---|---|---|---|
| `situ_and_mul` / `masked_situ_and_mul` | `csrc/libtorch_stable/activation_kernels.cu` | ✅ **应该能直接编译复用** | 纯 elementwise，无 arch 门控、无 CUTLASS/PTX 内联；已加入 `VLLM_STABLE_EXT_SRC` 常规列表。`forward_native` 给出了精确语义可做数值对照 |
| `concat_and_cache_mla_grouped` | `csrc/libtorch_stable/cache_kernels.cu` | ✅ 同上 | 与已有 `concat_and_cache_mla` 同族，多一层 cache 指针间接 |
| 6 × `fused_kimi_k3_mla_*` | `csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu` (1237 行) | ⚠️ **需验证**，无 arch 门控（进常规 `VLLM_STABLE_EXT_SRC`），但依赖 fp8 转换与 `fp8_ds_mla` 656B 布局 | **最大工作量所在**，详见 §3.1 |
| `fused_kda_decode` | `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` (1130 行) | ❌ 自动禁用 | 门控：CUDA≥13.0 ∩ `{9.0a,10.0f,12.0f}`，宏 `VLLM_ENABLE_FUSED_KDA_DECODE`；运行时还查 `hasattr(torch.ops._C,"fused_kda_decode")`。**PPU 上不编译、不调用，自动落 Triton** |
| `kimi_k3_attn_res` | `csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu` (954 行) | ❌ 自动禁用 | 门控：CUDA≥13.0 ∩ `{10.0f}`（仅 SM100），运行时还查 `is_device_capability_family(100)`。**自动落 Triton** |
| `custom_all_gather` / `custom_reduce_scatter` / `mnnvl_lamport_*` | `csrc/libtorch_stable/custom_all_gather_reduce_scatter.cu` 等 | ❌ 不需要 | `use_custom_allreduce() -> False`，SP 走 NCCL 回退 |
| `single_group_topk_block_kernel`（grouped topk 扩展） | `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` (+438) | ⚠️ 需验证 | 移植自 TRT-LLM `noAuxTcKernels.cu`，纯 warp/block shuffle，无 arch 内联；`fused_grouped_topk` 被 MegaMoE 路径调用（PPU 不走），非 MegaMoE 路径用 Python `grouped_topk`，所以**优先级低** |
| `dsv3_fused_a_gemm`（改造 +152） | `csrc/libtorch_stable/dsv3_fused_a_gemm.cu` | ⚠️ 我们已在用 | K3 的 `low_latency_gemm.py` 把它当 skinny GEMM 后端之一；PPU 上该 GEMM 选择表（按 SM103 实测）**不适用**，见 §3.3 |
| `merge_attn_states`（+39） | `csrc/libtorch_stable/attention/merge_attn_states.cu` | ✅ | chunked context 合并，我们已在用 |

### 2.2 Triton kernel（**这是复用价值最高的一块**）

| kernel | 文件 | PPU 判定 |
|---|---|---|
| KDA chunked prefill + fused gate | `vllm/models/kimi_k3/nvidia/ops/third_party/kda/{chunk,chunk_intra,chunk_intra_token_parallel}.py` (~2000 行) | ✅ 直接复用。`chunk_kda_with_fused_gate` 是 prefill 默认路径（`kda_prefill_backend` 解析后 flashkda 不可用即为 `triton`） |
| KDA recurrent decode / spec | `.../third_party/kda/fused_recurrent.py` (671 行) | ✅ 直接复用。`fused_recurrent_kda`（spec 多 query）、`fused_recurrent_kda_packed_decode`（纯 decode 回退）；**注意该文件用了 `is_arch_support_pdl()`** |
| KDA gate（`fused_kda_gate`、`fused_kda_gate_chunk_cumsum`） | 同上 | ✅ |
| `FusedRMSNormGated` | `vllm/third_party/flash_linear_attention/ops/fused_norm_gate.py` (+412) | ✅ 直接复用；也用 PDL |
| `gather_initial_states` | `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` (+83) | ✅ 直接复用；也用 PDL |
| `causal_conv1d_fn` / `causal_conv1d_update` | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py`（本 PR 改动 ±56） | ✅ 复用，但本 PR 新加了 `launch_pdl` 与 `do_not_specialize_on_alignment=["num_cache_lines"]` → 需验证 |
| **AttnRes** | `vllm/models/kimi_k3/nvidia/ops/attn_res.py` | ✅ 直接复用。原生 kernel 的 5 个前置条件里有 `is_device_capability_family(100)`，PPU 不满足 → 必走 Triton。该 Triton 版功能完整（含 delta / block write / output norm / online softmax） |
| **AttnRes (AMD 简版)** | `vllm/models/kimi_k3/amd/ops/attn_res.py` | 参考价值高：无 delta、无 output-norm 的最小实现，适合先跑通再对齐 |
| `_get_aligned_state_indices_kernel` / `_stage_spec_decode_metadata_kernel` | `vllm/models/kimi_k3/nvidia/kda_metadata.py` | ✅ 复用；`_metadata_launch_pdl()` 需验证 |
| `fused_mtp_input` | `vllm/models/kimi_k3/common/mtp.py` | ✅ 复用（embed+concat+RMSNorm） |
| `triton_merge_attn_states`（±54） | `vllm/v1/attention/ops/triton_merge_attn_states.py` | ✅ 已在用 |

**AMD 侧对我们的额外价值**：`vllm/models/kimi_k3/amd/` 整个目录（`linear.py` 1064 行 + `model.py` +
`mtp.py` + `ops/`）是上游**自己做的一次"非 NVIDIA 平台移植"**，它砍掉了：CuTe DSL tail fusion、
MegaMoE、fused MLA epilogue、low-latency GEMM 选择表、encoder CUDA graph、sequence parallel。
**这就是 PPU 移植的最佳蓝本 —— 先按 AMD 版的裁剪范围做 `ppu/`，再逐项加回。**

### 2.3 CuTe DSL kernel

| kernel | PPU 判定 |
|---|---|
| `latent_moe_tail/*`（`CollectiveKernel`、`AdaptiveUpProjectionKernel`、`LamportCopyKernel`，共 ~3400 行） | ❌ **放弃**。硬性要求 SM100 + TP∈{8,16} + bf16 + hidden 7168/latent 3584 + M≤16，且用 MNNVL multicast + Lamport 标志位。默认 `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=0`，**保持关闭即可** |
| `skinny_gemm.py` / `_skinny_gemm.py`（`ShapeDynamicSkinnyGemm`） | ❌ 短期放弃。需 `nvidia-cutlass-dsl`；`low_latency_gemm.py` 的形状表是 SM103 实测结果 |

### 2.4 外部扩展

| 扩展 | PPU 判定 |
|---|---|
| **FlashKDA**（`_flashkda_C`，CUTLASS） | ❌ 不可用。`flashkda.cmake` 匹配不到 arch 时 `add_custom_target(_flashkda_C)` 空目标；`resolve_kda_prefill_backend` 会因 `is_flashkda_supported()` 为 False 自动返回 `"triton"` |
| **DeepGEMM MegaMoE** | ❌ 不可用（PPU DeepGEMM 无 mega-moe API） |
| `flash-linear-attention==0.5.0`（新依赖） | ⚠️ 需确认是否要装（我们只用 in-tree 的 `vllm/third_party/flash_linear_attention`；这条依赖在 `requirements/cuda.txt`，PPU 用 `requirements/ppu.txt`） |
| `tilelang` 0.1.9 → **0.1.12** | ⚠️ 我们的 DSV4 mHC TileLang kernel 用 tilelang，**升级需回归**（见 `6191840b4` 那个 commit） |

---

## 3. 必做适配清单

### P0 —— 让 K3 能在 PPU 上被加载并跑通 eager

#### 3.0.1 建立 `vllm/models/kimi_k3/ppu/` 并接入平台分发

现状：`vllm/models/kimi_k3/__init__.py` 只有 rocm / else 两个分支，**PPU 会落到 `nvidia/`**，
从而 module-level 导入 `low_latency_gemm`（→ `cute_dsl/skinny_gemm`）和
`deepseek_v4/nvidia/ops/prepare_megamoe`。

改法（照抄 `deepseek_v4/__init__.py`）：

```python
if TYPE_CHECKING or current_platform.is_ppu():
    from .ppu.model import KimiK3ForConditionalGeneration, KimiLinearForCausalLM
    from .ppu.mtp import KimiK3MTP
elif current_platform.is_rocm():
    ...
```

`ppu/model.py` 建议**继承 NVIDIA 版**并关闭不可用能力（比 AMD 那样整份复制更易维护）：

- `KimiMoE`：强制 `use_mega_moe = False`
- `KimiDecoderLayer`：`use_sequence_parallel` 保持（NCCL 回退可用），但先建议关掉简化调试
- `MultiHeadLatentAttention`：替换 fused epilogue 为 PPU 版或 eager 回退（§3.1）
- `KimiK3DeltaAttention`：`decode_conv1d_weight`/`decode_norm_weight` 会因
  `is_fused_kda_decode_supported()` 为 False 而是 `None` → 自动走 Triton，**无需改动**
- `low_latency_gemm.enable_kimi_k3_low_latency_gemm(...)`：PPU 上直接 no-op（§3.3）

#### 3.0.2 `PPUPlatform.apply_config_platform_defaults` 加 K3 默认值

```python
# 建议在此处强制：
#   additional_config["kda_prefill_backend"] = "triton"       (显式，避免 auto 探测歧义)
#   envs.VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION 必须为 0
#   kernel_config.moe_backend != "deep_gemm_mega_moe" 时报清晰错误
```

同时考虑把 `KimiK3ForConditionalGeneration` 加进 V2 model runner 的排除名单
（`vllm/config/vllm.py` 的 `ROCM_EXCLUDED_V2_MODEL_RUNNER_ARCHITECTURES` 旁边加一个 PPU 版），
先用 V1 runner 跑通，避免同时调试两个变量。

#### 3.0.3 MoE：`PPUDeepGemmExpertsMXFP4` 开 SiTU

**这是 MoE 线的关键卡点**，两处必须改（`vllm/model_executor/layers/fused_moe/experts/ppu_deep_gemm_moe.py`）：

```python
@staticmethod
def _supports_activation(activation: MoEActivation) -> bool:
    return activation in [
        MoEActivation.SILU, MoEActivation.SWIGLUSTEP,
        MoEActivation.SWIGLUOAI, MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        # + MoEActivation.SITU          ← 需要加
    ]
```

`_act_mul_quant()` 调用 `self.activation(activation, act_out, input, clamp_limit=, alpha=, beta=)`；
而 `fused_moe/activation.py` 里 SITU 分支的签名是
`torch.ops._C.situ_and_mul(output, input, beta, linear_beta)`，
用的是 `activation_situ_beta` / `activation_situ_linear_beta`（来自 `FusedMoEConfig`），
**不是** `alpha/beta/clamp_limit` 那套 gpt-oss 参数。需要确认参数透传链路：
`KimiMoE → FusedMoE(activation_situ_beta=, activation_situ_linear_beta=) → FusedMoEConfig → experts`。

**量化方案决策（需要你拍板）**：`_supports_quant_scheme` 现在只认 `(kMxfp4Static, kMxfp4Dynamic)`
= W4A4。但 K3 的 checkpoint 是 **weight-only MXFP4（W4A16）**（上游 `_use_k3_situ_aiter`
的注释明确写了 "K3 is weight-only MXFP4 (W4A16)"，ROCm 走 `AITER_MXFP4_BF16`）。
三个选项：

| 选项 | 做法 | 代价 |
|---|---|---|
| A | 在 PPU 上按 W4A4 跑（激活也量化到 mxfp4） | 最省事，复用现有 `PPUDeepGemmExpertsMXFP4` 全部逻辑；**但偏离参考精度，必须做 eval 对比** |
| B | 给 `PPUDeepGemmExpertsMXFP4` 加 `(kMxfp4Static, None)` W4A16 分支，用 `m_grouped_bf16_gemm_nt_nopad` 之类 | 精度对齐，工作量中等 |
| C | 810E (sm80) 沿用现有 Marlin W4A16 路径，给 Marlin 加 SITU | 依赖 Marlin 的激活扩展性 |

注意我们分支现在的分流（`mxfp4.py`，rebase 后）：**非 sm80 → W4A4；sm80 → W4A16 + Marlin**。
所以 890P 默认落 A、810E 默认落 C。

#### 3.0.4 `situ_and_mul` 在 PPU 工具链下必须编出来

`csrc/libtorch_stable/activation_kernels.cu` 进的是常规 `VLLM_STABLE_EXT_SRC`，
理论上 PPU 编译会带上。验证方法：

```bash
python -c "import torch, vllm._C; print(hasattr(torch.ops._C, 'situ_and_mul'))"
```

若为 False，`SituAndMul.forward_cuda`（`is_cuda_alike()` 为 True 会走这条）会崩。
兜底：在 `SituAndMul.__init__` 的平台判断里对 PPU 加 `hasattr` 探测，缺失则退回 `forward_native`。

### P1 —— MLA 打通

#### 3.1 6 个 fused MLA epilogue 算子（最大工作量）

K3 的 `MultiHeadLatentAttention` **没有 eager/Triton 回退**：`_forward_prefill_fused` 与
`_decode_concat_cache` 无条件调用 `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py`
里的 4 个 Python 包装（背后 6 个 CUDA 算子）。

它们做的事其实很朴素：

| 算子 | 语义 |
|---|---|
| `fused_mla_key_concat_kv_cache_insert` | `k = cat(k_nope, k_pe)`（+可选 RoPE 应用到 `q`/`k_pe`）+ 把 `cat(kv_c_normed, k_pe)` 按 `slot_mapping` 写入 paged latent cache（bf16） |
| `fused_mla_key_concat_ds_mla_insert` | 同上，cache 是 `fp8_ds_mla` 656B per-tile self-scaled 布局 |
| `fused_mla_qkv_quant_kv_cache_fp8_insert` | 同上 + `q/k/v` 量化到 fp8（unscaled）+ latent 按 `_k_scale` 量化写入 |
| `fused_mla_decode_q_concat_kv_cache_insert`（3 变体） | `mqa_q = cat(ql_nope, q_pe)`（+可选 RoPE）+ 写 latent cache（bf16 / fp8 / ds_mla） |

**建议路线（从低风险到高性能）**：

1. **先写一个 PPU/通用 eager 回退**（纯 PyTorch：`torch.cat` + `rotary_emb` + 现有
   `concat_and_cache_mla`），放在 `vllm/models/kimi_k3/ppu/ops/fused_mla_key_concat_kv_cache.py`，
   由 `ppu/mla.py` 覆写导入。目标是**先让模型能出正确 token**。
   - PPU 已有的可复用件：`ops.concat_and_cache_mla`、`ops.concat_and_cache_mla_rope_fused`、
     `vllm/vllm_flash_attn/ppu/layers/rotary.py`、`vllm/models/deepseek_v4/common/ops/cache_utils.py`
2. **再验证 6 个 CUDA 算子能否在 PPU 上直接编译并数值正确** —— 它们没有 arch 门控，
   有很大概率能编。用 `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`（+221 行，PR 自带）
   直接跑对照。这一步如果通了，性能问题一次性解决。
3. **KV cache dtype 先只支持 bf16**。`fp8_ds_mla` 的 656B 布局和 plain-fp8 的 `_k_scale`
   路径都放到 P2；`mla.py` 里已有清晰的 `assert` 会告诉你哪条不支持。

#### 3.2 MLA backend 与 head 数

- K3 MLA 是 **NoPE-only**（`assert config.mla_use_nope`）+ **sigmoid output gate**（`g_proj`）。
  gate 是层内做的（`_gate_sigmoid_mul`，`torch.compile`），与 backend 无关 → 无移植成本。
- decode 走通用 `forward_mqa` → `FLASHMLA` 或 `TRITON_MLA`，两者都在 PPU 优先级表里。
  **PPU FlashMLA 支持任意 head 数**（`get_padded_num_q_heads` 里 PPU 分支直接返回 `num_heads`），
  比 NVIDIA 的 64/128 约束宽松，这是优势。
- prefill 走 `get_mla_prefill_backend(vllm_config)` → PPU 上应选到 `FLASH_ATTN`
  （`vllm/v1/attention/backends/mla/prefill/flash_attn.py`，已实现 `supports_out() -> True`）。
  **需要验证**：该 prefill backend 是否落到 `vllm/vllm_flash_attn/ppu/flash_attn_interface.py`，
  以及 `q_data_type` 为 bf16 时的 varlen 路径。
- 新增的 `non_causal_multi_token_decode`（MTP/DSpark 草稿组把非因果 query block 摊平成 decode 行）
  需要 backend 侧 `supports_non_causal_multi_token_decode = True`。**PPU FlashMLA/TritonMLA 默认 False**
  → MTP/DSpark 先关掉，P2 再开。

#### 3.3 `low_latency_gemm` 在 PPU 上必须 no-op

`vllm/models/kimi_k3/nvidia/low_latency_gemm.py` 的 `KIMI_K3_PROJECTIONS` 是
**GB300/SM103 上按 (N,K,M) 实测出来的 backend 选择表**（cute vs dsv3_fused_a），
对 PPU 完全无意义且会调用 CuTe DSL。

改法：`ppu/model.py` 里不调用 `enable_kimi_k3_low_latency_gemm`，或在该函数首行加
`if not current_platform.is_device_capability_family(100): return`。
后续如果要做 PPU 的 decode skinny GEMM 优化，应该另建一张 PPU 形状表，
复用 `vllm/utils/ppu_deep_gemm.py` 里的 `m_grouped_bf16_gemm_nt_nopad` 等。

### P2 —— 性能与功能补齐

| 项 | 说明 |
|---|---|
| KDA fused decode（PPU 版） | conv + KDA recurrence + gated norm 三合一。Triton 路径是 3 次 launch（`causal_conv1d_update` + `fused_recurrent_kda_packed_decode` + `o_norm`），decode 时 launch-bound 明显。参考 `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` 的融合边界 |
| AttnRes 融合 | Triton 版每层 2 次调用（pre-attn / post-attn），每次要读 `num_blocks+1` 个 hidden 向量。层数 × block 数增长时带宽压力大，值得做 PPU 原生 kernel。参考 `attn_res_kernel.cu` |
| Latent MoE tail | 不做 CuTe DSL 版，但**"两次 reduce 合成一次 all-reduce"这个思路是平台无关的** —— `LatentMoERunner._use_fused_path()` 的 concat+单次 all-reduce 已经是纯 PyTorch，PPU 直接受益，确认它被走到即可 |
| router gate / down-proj 多流 overlap | `maybe_execute_in_parallel` + `aux_stream()`，纯 PyTorch。PPU 上需确认多 stream 语义与 `torch.cuda.Event` 可用；阈值 `VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD` 需按 PPU 重新标定 |
| MTP / DSpark | 依赖 `non_causal_multi_token_decode`（§3.2）+ `concat_and_cache_mla_grouped` + `precompute_and_store_context_kv`。建议整体推到 P2 |
| 多模态（K2.5 ViT） | 我们已有 kimi_k25 路径。本 PR 改了 229 行（3D pos emb / packed patch merge / encoder CUDA graph / FA4 warmup）。encoder CUDA graph 与 `vision_fa4_warmup` 先关（`SupportsEncoderCudaGraph` 相关方法可覆写返回 None） |
| fp8 KV cache（plain + `fp8_ds_mla`） | 需要 §3.1 的 fp8 变体 + backend 的 `supports_quant_query_input` |

---

## 4. 已识别的具体陷阱（按踩坑概率排序）

1. **`current_platform.is_cuda()` 对 PPU 返回 True**。任何"只在 NVIDIA 跑"的守卫都会在 PPU 上放行。
   受影响的 K3 代码：`kimi_k3_triton_warmup()`（会真跑）、`_use_k3_situ_aiter()`（安全，先查 `is_rocm`）。
   写 PPU 分支时**要用 `is_ppu()` 显式判断，不要依赖 `is_cuda()` 为 False**。

2. **PDL（Programmatic Dependent Launch）**。`PPUPlatform` 继承 `CudaPlatform.is_arch_support_pdl()`
   → `major >= 9` 即 True，所以 **890P 上 PDL 会被打开**，本 PR 新增/改动的多个 Triton kernel
   会发 `tl.extra.cuda.gdc_wait()` / `gdc_launch_dependents()`：
   - `kimi_k3/nvidia/ops/attn_res.py`
   - `mamba/ops/causal_conv1d.py`（**本 PR 新加的**）
   - `mamba/ops/gather_initial_states.py`
   - `third_party/flash_linear_attention/ops/fused_norm_gate.py`
   - `kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py`
   - `kimi_k3/nvidia/kda_metadata.py`（`_metadata_launch_pdl()`）

   **好消息**：PDL 在我们分支上已被 PPU 路径实际使用（`deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`
   被 `deepseek_v4/ppu/ops/o_proj.py` 导入，`fused_moe/router/dsv4_topk.py` 也用），
   所以 PPU Triton 大概率支持这些 intrinsic。但**新增的 `causal_conv1d` PDL 分支必须单独验证** ——
   一旦不支持，最快的止血是给 `PPUPlatform` 覆写 `is_arch_support_pdl() -> False`。

3. **`do_not_specialize_on_alignment=["num_cache_lines"]`**（本 PR 给 `causal_conv1d` 两个 kernel 加的）
   —— 这是较新的 Triton 特性，PPU 的 Triton 分支若版本落后会直接 `TypeError`。

4. **`eager_break_during_capture`**。K3 的 `MultiHeadLatentAttention._attention` 与
   `KimiK3DeltaAttention._forward` 都带这个装饰器（breakable CUDA graph）。
   PPU 的 static graph wrapper 继承 `CUDAGraphWrapper`，需确认 breakable capture 语义可用
   （`tests/v1/cudagraph/test_breakable_cudagraph.py` 是 PR 自带的 48 行测试）。

5. **`tilelang` 0.1.9 → 0.1.12**（`requirements/cuda.txt`）。我们的 DSV4 mHC TileLang kernel
   刚在 `6191840b4` 修过 PPU 分支，升级需回归。PPU 用 `requirements/ppu.txt`，
   确认那边的 tilelang pin 是否需要同步。

6. **`vllm/config/parallel.py` 删掉了 `nnodes > 1 → disable_custom_all_reduce`**。
   PPU 因 `use_custom_allreduce() -> False` 不受影响，但如果将来打开 custom AR，多节点会踩。

7. **`vllm/platforms/interface.py` 把 MLA 模型的 `kernel_block_alignment_size` 抬到 ≥128**。
   PPU 继承此逻辑 ⇒ **K3（以及我们现有的 DSV4）在 PPU 上的 KV manager block size 会变**，
   需确认 PPU FlashMLA 的 block 约束兼容 128 对齐。

8. **`vllm/v1/core/kv_cache_utils.py` 的分组合并变化 + `scheduler.py` 的 mamba-align sub-block 推进**
   会影响所有 hybrid 模型 —— K3 是 MLA + KDA(mamba state) 混合，正是这条路径的重度用户。

9. **MXFP4 派发行为变化**（rebase 报告 §2.1）：非 GPT-OSS 的 mxfp4 checkpoint 现在统一走
   `Mxfp4MoEMethod`。我们原来的 `is_ppu() and not GptOss` 判断已被 `_make_moe_method` 多态取代 ——
   如果 PPU 上出现 mxfp4 模型行为变化，先查这里。

10. **`vllm/models/kimi_k3/__init__.py` 目前会让 PPU 落进 `nvidia/`** —— 在 §3.0.1 完成之前，
    任何 PPU 上加载 K3 的尝试都会在 import 阶段就带进 CuTe DSL / MegaMoE 依赖。

---

## 5. 建议的落地顺序与验收点

| 阶段 | 内容 | 验收 |
|---|---|---|
| **S0** | lint + 纯 Python 导入冒烟；PPU 上完成一次全量编译；确认 3 个新算子在/不在 `torch.ops._C` | `hasattr(torch.ops._C,'situ_and_mul')==True`；`fused_kda_decode`/`kimi_k3_attn_res` 为 False（预期）；`_flashkda_C` 缺失（预期） |
| **S1** | 回归现有 PPU 模型（DSV4、GPT-OSS MXFP4） | 精度与吞吐无退化（因为动了 MXFP4 派发 + KV 分组 + 调度器） |
| **S2** | 单 kernel 对数：`tests/models/kimi_k3/test_attn_res.py`、`test_kda.py`、`test_kda_metadata.py`、`tests/kernels/core/test_activation.py`（SiTU） | Triton 路径全绿；KDA fused decode 用例按 `is_fused_kda_decode_supported` 自动 skip |
| **S3** | `vllm/models/kimi_k3/ppu/` 骨架 + 平台分发 + `low_latency_gemm` no-op + eager MLA epilogue 回退 | K3 能在 PPU 上以 `--enforce-eager` 加载并产出通顺文本（TP=1 起） |
| **S4** | 6 个 fused MLA 算子在 PPU 上编译验证：`tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py` | 通过则去掉 eager 回退 |
| **S5** | MoE：`PPUDeepGemmExpertsMXFP4` 开 SiTU + 量化方案定案 | `tests/kernels/moe/` 相关 + 端到端 eval（W4A4 vs W4A16 精度对比是**必须交付项**） |
| **S6** | TP / EP / sequence parallel（NCCL 回退）扩到目标规模 | `tests/models/kimi_k3/test_sequence_parallel.py` 可作参照 |
| **S7** | CUDA graph（含 breakable）、多流 overlap、性能 kernel（KDA fused decode / AttnRes 原生） | `vllm bench` |
| **S8** | MTP / DSpark / 多模态 | — |

---

## 6. 关键文件速查

**要改的（PPU 新增/修改）**

```
vllm/models/kimi_k3/__init__.py                       ← 加 is_ppu() 分支
vllm/models/kimi_k3/ppu/{__init__,model,mla,mtp}.py   ← 新建（继承 nvidia/，裁剪不可用能力）
vllm/models/kimi_k3/ppu/ops/fused_mla_key_concat_kv_cache.py  ← 新建（eager/PPU 回退）
vllm/model_executor/layers/fused_moe/experts/ppu_deep_gemm_moe.py  ← 开 SiTU + 量化方案
vllm/platforms/ppu.py                                 ← apply_config_platform_defaults 加 K3 默认
vllm/models/kimi_k3/nvidia/low_latency_gemm.py        ← 加 arch 守卫（或 ppu 侧不调用）
vllm/config/vllm.py                                   ← （可选）PPU 也排除 K3 的 V2 runner
```

**要读的（复用来源）**

```
vllm/models/kimi_k3/amd/                              ← 最佳移植蓝本（上游自己做的非 NV 移植）
vllm/models/deepseek_v4/ppu/                          ← PPU 平台隔离范式（model.py 仅 73 行）
vllm/models/kimi_k3/nvidia/ops/third_party/kda/       ← KDA Triton，直接复用
vllm/models/kimi_k3/nvidia/ops/attn_res.py            ← AttnRes Triton，直接复用
vllm/utils/ppu_deep_gemm.py                           ← PPU DeepGEMM 全部可用 API
vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py  ← Latent MoE 编排（平台无关）
tests/models/kimi_k3/                                 ← PR 自带的对数测试
tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py       ← 6 个 MLA 算子的对照
```

**明确不做的**

```
vllm/models/kimi_k3/nvidia/ops/cute_dsl/              ← SM100 + MNNVL，放弃
vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py       ← 放弃
csrc/libtorch_stable/custom_all_gather_reduce_scatter*           ← use_custom_allreduce=False
cmake/external_projects/flashkda.cmake                ← arch 不匹配，空 target
KimiK3MegaMoEExperts / deep_gemm_mega_moe             ← PPU DeepGEMM 无对应 API
```
