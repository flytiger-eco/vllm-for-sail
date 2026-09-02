# Kimi K3 上游适配分析报告（vllm-project/vllm PR #50000）

> 报告日期：2026-07-27
> 分析对象：<https://github.com/vllm-project/vllm/pull/50000> — `[New model] Kimi K3`
> 作者：ZJY0516 (Jiangyun Zhu)，分支 `vllm-project/vllm:kimi-k3`
> 规模：**263 files changed, +39,811 / −2,390**，基于上游 `main`（`0ba2aa35a`）
> 本地分析分支：`feat/kimi-k3-ppu`（fork 自 `rebase/v0.26.0`）

---

## 0. 一句话总结

PR #50000 不是一次"加个 modeling 文件"的常规模型适配，而是一次**带自研 kernel 栈的整体落地**：
Kimi K3 = `MLA (NoPE + output gate) + KDA 线性注意力`混合骨干 + `Latent MoE`（专家在
低维 latent 空间计算）+ `AttnRes`（跨层残差流 softmax 混合）+ `SiTU` 激活 + MXFP4 专家权重
+ K2.5 ViT 多模态 + MTP/DSpark 双投机头。为此上游新增了 **6 个 libtorch-stable CUDA 算子、
2 个独立 arch-gated CUDA kernel、1 个外部 FlashKDA 扩展、5 个 CuTe DSL kernel、
约 10 个 Triton kernel**，并顺带把 custom all-reduce 扩成了 all-gather / reduce-scatter
（含 MNNVL Lamport 路径）。

---

## 1. 模型结构（从代码反推）

### 1.1 配置入口

| 配置类 | 文件 | 说明 |
|---|---|---|
| `KimiK3Config` | `vllm/transformers_utils/configs/kimi_k3.py` | 顶层多模态壳：`text_config` (=`KimiLinearConfig`) + `vision_config` (=`KimiK3VisionConfig`) |
| `KimiLinearConfig`（扩展） | `vllm/transformers_utils/configs/kimi_linear.py` | 新增 7 个字段，见下表 |
| `K3DSparkConfig` | `vllm/transformers_utils/configs/k3_dspark.py` | DSpark 草稿模型（dense MLA，继承 `DeepseekV2Config`） |

`KimiLinearConfig` 新增字段决定了本次全部新算子的开关：

| 字段 | 作用 |
|---|---|
| `attn_res_block_size` | 非 `None` → 启用 **AttnRes** 跨层残差流机制 |
| `routed_expert_hidden_size` | 非 `None` → 启用 **Latent MoE**（专家 hidden 维 < 模型 hidden 维） |
| `latent_moe_use_norm` | Latent MoE 输出上投影前是否插 RMSNorm |
| `activation_situ_beta` / `activation_situ_linear_beta` | **SiTU** 激活的两个 β |
| `mla_use_output_gate` | MLA 输出 sigmoid 门（`g_proj`） |
| `topk_method` (`noaux_tc`) | 路由方式（grouped topk + e_score_correction_bias） |
| `max_position_embeddings` | — |

关键事实：**目标模型 hidden_size = 7168，latent (routed_expert_hidden_size) = 3584**
（`latent_moe_tail.py` 硬断言），MoE `moe_intermediate = 3072`（AMD 注释），KDA `head_dim = 128`。

### 1.2 层拓扑

```
KimiDecoderLayer(layer_idx)
├── is_kda_layer(layer_idx)?
│   ├── use_full_rank_gate=True  → KimiK3DeltaAttention   (K3 专用 KDA)
│   └── else                     → KimiGatedDeltaNetAttention (兼容 Kimi-Linear)
└── else                         → MultiHeadLatentAttention (K3 专用 MLA，NoPE-only)
    ↓
├── is_moe_layer? → KimiMoE (Latent MoE + shared experts)
└── else          → KimiMLP  (SiluAndMul 或 SituAndMul)
```

`linear_attn_config` 里的 `kda_layers` / `full_attn_layers` 决定混合排布 —— 与
Kimi-Linear 一致，K3 只是把 KDA 换成 full-rank gate 变体。

### 1.3 AttnRes：本 PR 最"新"的结构创新

普通 pre-norm Transformer 是 `h = h + f(norm(h))`。K3 换成了一组**残差块（residual
streams）+ softmax 混合**：

- 每层维护 `blocks: [num_tokens, num_blocks, hidden]` 与 `prefix_sum: [num_tokens, hidden]`；
- 每 `attn_res_block_size` 层把当前 `prefix_sum` 快照写入 `blocks[block_write_idx]`；
- 进入 attn/MLP 前，对 `num_blocks + 1` 个候选残差源分别做 RMSNorm，
  用 `norm_weight * qk_weight`（`*_res_proj` 是一个 `hidden → 1` 的线性层）算 logit，
  在源维度上做 online softmax，得到加权混合向量，再做一次输出 RMSNorm 喂给子层。

即"**在残差流上做一次单头注意力**"。参考实现见 `vllm/models/kimi_k3/nvidia/ops/attn_res.py`
的 Triton kernel（`_attn_res_kernel`），数学上等价于：

```
logits_i = rmsnorm(src_i) · (norm_w * qk_w)        # i ∈ [0, num_blocks]
mixed    = softmax_i(logits_i) · src_i             # online softmax，含 max 重标定
out      = rmsnorm(mixed) * output_norm_w          # 可选
```

`num_blocks == 0` 时 softmax 恒为 1，直接返回 prefix。

### 1.4 Latent MoE

`hidden(7168) --routed_expert_down_proj--> latent(3584) --experts(MXFP4, SiTU)--> latent
--[RMSNorm]--> --routed_expert_up_proj(replicated)--> hidden(7168) --+ shared_experts(hidden)`

要点：
- `down_proj` / `up_proj` 都是 **`ReplicatedLinear`（未量化）**。up_proj 复制是为了让
  `LatentMoERunner` 把「latent 部分和 shared 部分的两次 reduce」拼成**一次 all-reduce**：
  concat 未 reduce 的 latent partial (d) 与 shared partial (D) → 单次 all-reduce → split
  → 本地 norm + up-proj，`shared` 加法折进 `torch.addmm` epilogue。
- `down_proj` 被移出 `FusedMoE`，与 router gate 放到**两条 CUDA stream 上并行**
  （`maybe_execute_in_parallel`，阈值 `VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD=256`）。
- 专家本体两条路：
  1. `moe_backend == "deep_gemm_mega_moe"` → `KimiK3MegaMoEExperts`（继承
     `DeepseekV4MegaMoEExperts`，走 DeepGEMM `fp8_fp4_mega_moe` + 对称显存 buffer，
     **强制要求 EP、SiTU、latent MoE、grouped-topk、单 expert group**）；
  2. 否则 `FusedMoE(runner_cls=LatentMoERunner)`。

### 1.5 MLA（K3 版）

`vllm/models/kimi_k3/nvidia/mla.py` 是一个**自持有 KV cache 的 `AttentionLayerBase`**
（不走 `MLAAttention.forward` 编排），显式分裂 prefill / decode：

- **prefill**：`kv_b_proj` 得到 `k_nope|v` → 一个 fused kernel 同时完成
  `k = concat(k_nope, k_pe)`（可选 RoPE）+ latent 写 paged cache；按 cache dtype 分派
  bf16 / plain-fp8 / `fp8_ds_mla`（656B per-tile self-scaled）三条 kernel；
  然后 `prefill_backend.run_prefill_new_tokens(..., out=)`，chunked context 用
  `merge_attn_states` online-softmax 合并。
- **decode**：`BMM1` 用 `W_UK_T` 把 `q_nope` 吸收进 latent 空间 → fused
  `concat(ql_nope, q_pe)` + latent 写 cache → `impl.forward_mqa` → `W_UV` BMM 上投影。
- **output gate**：`g_proj(hidden).sigmoid() * attn_out`，小 batch（<512 token）时
  `g_proj` GEMM 放 aux stream 与注意力前端 overlap。
- NoPE-only（`assert config.mla_use_nope`）；DSpark 草稿模型复用同一个类但开 RoPE。

### 1.6 KDA（Kimi Delta Attention）

`vllm/models/kimi_k3/nvidia/kda.py`，继承 `GatedDeltaNetAttention`：

- 输入投影 `in_proj_qkvgfab`：一次 GEMM 出 `q,k,v,g(各 H*D) + f_a(D) + beta(H)`，
  并对 TP-local 行做 16 对齐 padding 以命中对齐 BF16 GEMM；
- gate 全秩：`f_b_proj: head_dim → H*head_dim`（这是 `use_full_rank_gate` 的含义）；
- `conv1d` 打包 Q/K/V 三路（`3 * projection_size`，fp32 权重），另存一份
  **width-major 副本 `decode_conv1d_weight`** 专供 fused decode kernel；
- `o_norm = FusedRMSNormGated(head_dim, activation="sigmoid")`，decode 时额外保存
  fp32 `decode_norm_weight`；
- 三条执行路径：
  | 场景 | kernel |
  |---|---|
  | 纯 decode、无 spec、满足 arch/dtype 约束 | **`ops.fused_kda_decode`**（conv + KDA recurrence + gated norm 全融合，单 launch） |
  | prefill | `flashkda`（外部 CUDA 扩展）或 Triton `chunk_kda_with_fused_gate` |
  | spec decode（多 query） | `causal_conv1d_update` + Triton `fused_recurrent_kda` |
  | 回退 decode | `causal_conv1d_update` + Triton `fused_recurrent_kda_packed_decode` |

`fused_kda_decode` 的启用条件（`is_fused_kda_decode_supported`）非常窄：
`num_heads ∈ {12,24,48,96}`、`head_dim == 128`、`conv_width == 4`、`num_spec == 0`、
bf16 输入与 conv state、非 `conv_state_dim_first`、SM90 / SM10x / SM12x。

### 1.7 多模态与投机解码

- **ViT**：复用 `kimi_k25_vit`（`MoonViT3dPretrainedModel` + `KimiK25MultiModalProjector`），
  本 PR 对其做了 229 行改动：新增 `get_pos_embeds` 3D 位置嵌入、`_make_vision_norm`、
  `build_image_merge_gather_idx` / `tpool_patch_merger_packed`（打包 patch merge）、
  AMD 上走 `aiter.ops.triton.conv.conv2d`、以及 encoder CUDA graph 支持
  (`SupportsEncoderCudaGraph`, `prepare_encoder_cudagraph_metadata`)。
- **MTP**：`KimiK3MTP`（`method="kimi_k3_mtp"`），`common/mtp.py` 提供 Triton
  `fused_mtp_input`（embed + hidden 拼接 + RMSNorm 融合）。
- **DSpark**：`K3DSparkModel` / `K3DSparkForCausalLM`，dense MLA 草稿头，带
  `precompute_and_store_context_kv`（预算 context KV 并直接写 draft cache）+
  `ReplicatedDSparkMarkovHead`。要求 `decode_context_parallel_size == 1`。

---

## 2. 新增 / 变更 kernel 全清单

### 2.1 libtorch-stable CUDA 算子（`csrc/libtorch_stable/`，随 `_C_stable_libtorch` 编译）

| 注册名 | 源文件 | 作用 |
|---|---|---|
| `fused_kimi_k3_mla_key_concat_kv_cache_insert` | `fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu` (+1237) | prefill：`k=concat(k_nope,k_pe)` + 可选 RoPE + latent 写 paged cache（bf16） |
| `fused_kimi_k3_mla_key_concat_ds_mla_insert` | 同上 | 同上，cache 为 `fp8_ds_mla` 656B 布局 |
| `fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert` | 同上 | 同上 + q/k/v 量化到 fp8（per-tensor） |
| `fused_kimi_k3_mla_decode_q_concat_kv_cache_insert` | 同上 | decode：`mqa_q=concat(ql_nope,q_pe)` + 写 cache（bf16） |
| `fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert` | 同上 | 同上，fp8 cache + fp8 query |
| `fused_kimi_k3_mla_decode_q_concat_ds_mla_insert` | 同上 | 同上，`fp8_ds_mla` |
| `situ_and_mul` / `masked_situ_and_mul` | `activation_kernels.cu` (+108) | SiTU GLU：`β·tanh(g/β)·sigmoid(g) · [β_l·tanh(u/β_l)]`；masked 版供 MoE batched 路径 |
| `concat_and_cache_mla_grouped` | `cache_kernels.cu` (+96) | 多 cache 指针（分组）版 MLA concat-and-cache，DSpark 用 |
| `custom_all_gather` / `custom_reduce_scatter` / `mnnvl_lamport_all_gather` / `mnnvl_lamport_reduce_scatter` | `custom_all_gather_reduce_scatter.cu` (+362)、`custom_all_gather_reduce_scatter.cuh` (+326)、`custom_collective_common.cuh` (+332) | 把原 `custom_all_reduce` 拆出公共部分，新增 AG/RS + MNNVL multicast Lamport 路径 |
| `dsv3_fused_a_gemm`（改造） | `dsv3_fused_a_gemm.cu` (+152/−?) | 被 K3 low-latency GEMM 复用为 skinny GEMM 后端之一 |

### 2.2 arch-gated 独立 CUDA kernel（编译期宏 + `cuda_archs_loose_intersection`）

| 算子 | 源文件 | 编译条件 | 运行时条件 |
|---|---|---|---|
| `fused_kda_decode` | `csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu` (+1130) | CUDA ≥ 13.0，archs `9.0a;10.0f;12.0f`，`--use_fast_math`，宏 `VLLM_ENABLE_FUSED_KDA_DECODE` | 见 §1.6 |
| `kimi_k3_attn_res` | `csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu` (+954) | CUDA ≥ 13.0，arch **`10.0f` only**，`--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`，宏 `VLLM_ENABLE_KIMI_K3_ATTN_RES` | `hidden_size==7168` 且 有 delta 且 有 output_norm 且 `num_blocks>0` 且 `block_write_idx<0` 且 SM100 family |

两者都有 **Triton 回退**，这对非 NVIDIA 后端是极好的消息（见交接报告）。

### 2.3 外部 CUDA 扩展：FlashKDA

- `cmake/external_projects/flashkda.cmake` + `csrc/flashkda_registration.cpp`
- FetchContent 拉 `https://github.com/vllm-project/FlashKDA.git @ a3e42bbb`（带 cutlass 子模块）
- 产出独立扩展 `vllm/_flashkda_C.abi3.so`，注册 `torch.ops._flashkda_C.fwd` /
  `get_workspace_size`
- 支持 arch：CUDA≥12.0 → `9.0a`；CUDA≥13.0 → `10.0f,12.0f`；12.9 → `10.0a,10.3a,12.0a`
- 仅用于 **KDA prefill**（chunked 形式），可用 `additional_config.kda_prefill_backend`
  在 `auto|triton|flashkda` 间切换

### 2.4 CuTe DSL kernel（Python，`nvidia-cutlass-dsl`）

| 文件 | 行数 | 作用 |
|---|---|---|
| `.../ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py` | 1012 | `CollectiveKernel`：融合 all-reduce + RMSNorm + reduce-scatter，带 early-exit |
| `.../latent_moe_tail/fused_add_multicast_gemm.py` | 1291 | `AdaptiveUpProjectionKernel` 动态分支：up-proj GEMM + shared add + multicast 写出 |
| `.../latent_moe_tail/fused_add_multicast_skinny_gemm.py` | 438 | 同上的 skinny（M ≤ 5）分支 |
| `.../latent_moe_tail/lamport_copy.py` | 226 | `LamportCopyKernel`：Lamport 标志位收敛拷贝 |
| `.../latent_moe_tail/primitives.py` | 437 | 共享 primitives |
| `vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py` + `_skinny_gemm.py` | 252+180 | 通用 BF16 skinny GEMM（`ShapeDynamicSkinnyGemm`），K3 decode 用 |

**Latent MoE tail fusion** 整体门槛极高：`SM100` + `TP ∈ {8,16}` + `bf16` +
`hidden=7168 / latent=3584` + `1 ≤ M ≤ 16`，且不支持 DBO/ubatching/sleep mode。
由 `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` 显式开启（默认关）。

### 2.5 Triton kernel

| 文件 | 内容 |
|---|---|
| `vllm/models/kimi_k3/nvidia/ops/third_party/kda/{chunk,chunk_intra,chunk_intra_token_parallel,fused_recurrent}.py` | ~2346 行，KDA 的 chunked 前向 + fused gate + 递归 decode（`chunk_kda_with_fused_gate`、`fused_recurrent_kda`、`fused_recurrent_kda_packed_decode`、`fused_kda_gate`…），来自 flash-linear-attention，pre-commit 已排除 |
| `vllm/models/kimi_k3/amd/ops/third_party/kda/*` | ~2415 行，AMD 版（含 `chunk_intra_token_parallel` 的 ROCm 变体） |
| `vllm/models/kimi_k3/nvidia/ops/attn_res.py` | AttnRes Triton 参考实现（含 PDL `gdc_wait`/`gdc_launch_dependents`） |
| `vllm/models/kimi_k3/amd/ops/attn_res.py` | AttnRes AMD Triton 版（无 delta / 无 output-norm 的简化签名） |
| `vllm/third_party/flash_linear_attention/ops/fused_norm_gate.py` | +412，`FusedRMSNormGated`（sigmoid gated RMSNorm） |
| `vllm/model_executor/layers/mamba/ops/gather_initial_states.py` | +83，按 state index gather 初始 recurrent state |
| `vllm/models/kimi_k3/common/mtp.py` | `fused_mtp_input`：embed+hidden concat+RMSNorm |
| `vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py` | +244，上述 6 个 CUDA 算子的 Python 包装 + Triton 回退 |
| `csrc/libtorch_stable/moe/grouped_topk_kernels.cu` (+438) / `moeTopKFuncs.cuh` (±309) | `single_group_topk_block_kernel`，移植自 TRT-LLM `noAuxTcKernels.cu`，支撑 `fused_grouped_topk` |

### 2.6 DeepGEMM MegaMoE（外部依赖能力，非本 PR 新写）

`KimiK3MegaMoEExperts` 依赖 DeepGEMM 侧这些 API：
`transform_sf_into_required_layout`、`transform_weights_for_mega_moe`、
`get_symm_buffer_for_mega_moe`、`fp8_fp4_mega_moe(..., activation="situ",
activation_beta=, activation_linear_beta=, activation_clamp=, fast_math=)`。
`tilelang` 也从 0.1.9 升到 **0.1.12**；新增依赖 `flash-linear-attention==0.5.0`。

---

## 3. 框架层改动（与 K3 强相关但影响面更广）

| 区域 | 改动 | 备注 |
|---|---|---|
| **模型注册** | `KimiK3ForConditionalGeneration`、`KimiK3MTPModel`、`K3DSparkModel` 入 registry；`KimiLinearForCausalLM` 从 `vllm/model_executor/models/kimi_linear.py`（**删除 646 行**）迁到 `vllm.models.kimi_k3` | 平台隔离目录：`kimi_k3/{nvidia,amd,common}` |
| **量化路由** | `KimiK3ForConditionalGenerationConfig`：把 checkpoint 的 `compressed-tensors + mxfp4-pack-quantized` **改写成 `quant_method="mxfp4"`**，从 `CompressedTensorsW4A4Mxfp4MoEMethod` 换到 `Mxfp4MoEMethod`（更宽的 backend 集合）；同时补 patch `model_arch_config.quantization_config` | 对 PPU 直接相关，见交接报告 |
| **MXFP4 后端** | `Mxfp4Config._make_moe_method` 抽成可覆写；新增 `_use_k3_situ_aiter`（ROCm gfx950 + SiTU + AITER）；`_setup_kernel_k3_situ` 处理 separated vs gate/up-interleaved 权重布局；新 env `AITER_SITUV2_A8W4` | Marlin / TRT-LLM mxfp4/nvfp4 / rocm_aiter_moe 均有配套改动 |
| **MoE runner** | 新 `LatentMoERunner`；`MoERunner` 抽出 `_unpack`、`routed_output_transform`、`_fused_output_is_reduced` 等 hook | |
| **MLA 非因果多 token decode** | `MLAAttentionSpec.non_causal_multi_token_decode`；builder 侧 `supports_non_causal_multi_token_decode`；`kv_cache_utils` 新增"同类型层按 spec 可 merge 进一步合并分组" | 让 draft/target 头数不同、causal 属性不同的 MLA 层能共组 |
| **调度器** | mamba align 模式允许 sub-block 进度并在下一个可缓存位置重新对齐（`test_partial_prefix_cache_hits.py`） | 移除了 `block_size <= max_num_batched_tokens` 的硬断言 |
| **V2 runner 默认** | `default_v2_model_runner_architectures()`：ROCm 上把 `KimiK3ForConditionalGeneration` 排除出 V2 默认（profile run 会 fault） | |
| **多节点 custom AR** | 删除 `nnodes > 1 → disable_custom_all_reduce`，改由 MNNVL 能力探测决定 | |
| **Tokenizer / Renderer / Parser（Rust）** | `tokenizer_mode="kimi_k3"`：用 HF tokenizer 但走 **Python XTML 编码**而非 Jinja 模板；`rust/src/chat/src/renderer/kimi_k3/encoding.rs` (+568)、`rust/src/parser/src/unified/kimi_k3.rs` (+1149) + `structural_tag.rs` (+500)；Python 侧 `vllm/renderers/kimi_k3.py`、`vllm/reasoning/kimi_k3_reasoning_parser.py` (+354)、`vllm/tool_parsers/kimi_k3_tool_parser.py` (+405) | Cargo 依赖切到 public `llm-multimodal` |
| **结构化输出** | `structural_tag_registry.py` (+263)、`v1/structured_output/__init__.py` (+66)、MTP 结构化输出测试 | |
| **KV connector (NIXL)** | `base_worker.py` (+261)、HMA（异构内存）支持、desc geometry 测试 (+573)、TP mapping | 与 K3 混合 KV（MLA + mamba state）相关 |
| **Warmup** | `kimi_k3_triton_warmup.py`（+182，预热 AttnRes/KDA Triton profile）替换 `qwen_triton_warmup.py`（−114）；`vision_fa4_warmup.py`（+195，ViT FA4 编译预热）；`v1_block_table_warmup.py`、`cutedsl_warmup` provider 注册 | |
| **CUDA graph** | `breakable_cudagraph`：`@eager_break_during_capture` 用在 MLA `_attention` 与 KDA `_forward` 上 | |

---

## 4. 环境变量与开关

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION` | `0` | 开启 CuTe DSL latent-MoE tail 融合（SM100 + TP8/16 + bf16） |
| `VLLM_ROUTED_DOWN_PROJ_STREAM_TOKEN_THRESHOLD` | `256` | router gate 与 routed down-proj 多流 overlap 的 token 上限 |
| `AITER_SITUV2_A8W4` | `0` | ROCm：SiTU MXFP4 走 a8w4 gate/up-interleaved flydsl kernel |
| `VLLM_MARLIN_MXFP8_INPUT_QDQ` | `0` | 调试：在 W4A16 Marlin 上模拟 W4A8 |
| `additional_config.kda_prefill_backend` | `auto` | `auto` / `triton` / `flashkda` |
| `--kernel-config moe_backend=deep_gemm_mega_moe` | — | 启用 MegaMoE（要求 `--enable-expert-parallel`） |

---

## 5. 测试面

新增/改动测试 ~30 个文件，值得注意的：

- `tests/models/kimi_k3/test_attn_res.py` / `test_amd_attn_res.py`：AttnRes 数值对齐
- `tests/models/kimi_k3/test_kda.py` (+757) / `test_kda_metadata.py` (+411)：替换旧
  `tests/kernels/test_kda.py`（−226）
- `tests/models/kimi_k3/test_latent_moe_tail.py`、`test_sequence_parallel.py`
- `tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py`：6 个 fused MLA 算子
- `tests/kernels/test_bf16_skinny_gemm.py` (+657)
- `tests/kernels/moe/test_grouped_topk.py` (+266)
- `tests/v1/attention/test_mla_noncausal.py`、`tests/v1/cudagraph/test_breakable_cudagraph.py`
- Rust/parser/renderer 侧一整套 fixture 驱动测试

---

## 6. 对我们（PPU）的第一层判断

| 组件 | 可移植性 | 理由 |
|---|---|---|
| AttnRes | **高** | 有完整 Triton 参考实现（`nvidia/ops/attn_res.py` 与更简的 `amd/ops/attn_res.py`），CUDA 版仅 SM100 |
| KDA prefill / spec / 回退 decode | **高** | 全 Triton（`third_party/kda/*`），AMD 已有一份可直接对照的移植 |
| KDA fused decode | 低（需自研） | 纯 CUDA，SM90/10x/12x，`--use_fast_math`；回退到 Triton 路径即可先跑通 |
| FlashKDA prefill | 低（需自研或跳过） | 外部 CUTLASS 扩展；`kda_prefill_backend=triton` 可绕过 |
| MLA 6 个 fused epilogue 算子 | 中 | 逻辑简单（concat + 可选 RoPE + 写 cache + 可选量化），但布局/量化细节多；可用 PPU 已有 `concat_and_cache_mla` 类算子逐步替换 |
| SiTU 激活 | **高** | 小 elementwise kernel，`forward_native` 已给出精确语义 |
| Latent MoE（非 tail-fusion） | **高** | 纯 Python 编排 + `FusedMoE`/`LatentMoERunner`，PPU 只需专家 kernel 支持 SiTU |
| MegaMoE (`fp8_fp4_mega_moe`) | 中 | 依赖 DeepGEMM 侧 mega-moe API；我们已有 `ppu_deep_gemm` |
| Latent MoE tail fusion (CuTe DSL) | 极低 | SM100 专属 + MNNVL multicast + Lamport；直接关掉 |
| custom AG/RS + MNNVL Lamport | 低 | 依赖 CUDA IPC / multicast；SP 有 NCCL 回退（`sp_all_gather` → `tensor_model_parallel_all_gather`） |
| ViT (K2.5) | 中 | 我们已有 kimi_k25 路径可参考；encoder CUDA graph 与 FA4 warmup 可先关 |
| Rust renderer/parser/tokenizer | **高（平台无关）** | 纯 CPU 逻辑，直接复用 |

详细的 PPU 适配清单见 `03-ppu-handoff.md`。

---

## 7. 参考文件索引（按重要性）

```
vllm/models/kimi_k3/nvidia/model.py            1859  层编排 / MoE / decoder layer / 多模态壳
vllm/models/kimi_k3/nvidia/mla.py               763  自持 KV cache 的 MLA
vllm/models/kimi_k3/nvidia/kda.py               698  KDA layer + 后端选择
vllm/models/kimi_k3/nvidia/kda_metadata.py      496  KDA metadata builder（含 2 个 Triton staging kernel）
vllm/models/kimi_k3/nvidia/mtp.py               443  MTP
vllm/models/kimi_k3/nvidia/dspark_mla.py        526  DSpark 草稿头
vllm/models/kimi_k3/nvidia/low_latency_gemm.py  517  按 (N,K,M) 查表选 skinny GEMM 后端
vllm/models/kimi_k3/common/mm_preprocess.py     429  多模态 processor / dummy inputs
vllm/model_executor/layers/fused_moe/runner/latent_moe_runner.py  255
csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu          1130
csrc/libtorch_stable/kimi_k3/attn_res_kernel.cu                   954
csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu 1237
```
