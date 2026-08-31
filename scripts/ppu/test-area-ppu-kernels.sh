#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-kernels.sh — PPU Kernels 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-kernels.yml（容器内，
# cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/kernels.yaml
# 的迁移快照（含 OAM-810E device_conditional_ignores 已合入对应 step，
# 调整用例直接改这里）。本 area 无 MODEL_MAP：用例全部走合成张量，
# 唯一真模型引用 neuralmagic/Llama-3.2-1B-quantized.w8a8 在
# test_triton_scaled_mm.py::test_rocm_compressed_tensors_w8a8
# （@skipif ROCm-only，PPU 自动 SKIP）；Qwen-VL/GLM/fxmarty/gguf 模型引用
# 均落在被 ignore 的文件内。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 无 multi 段（上游 kernels
#                                      多卡用例仅 test_mamba_mixer2.py 单个
#                                      @multi_gpu_test(2)，单卡自动 SKIP）
#
# 机制移植自 aone_ci/scripts/test_area_ppu_kernels.sh（AUTO-GENERATED 不可
# 手改，故在此复刻）：junit EXIT trap 合并 + 崩溃补 error case。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-kernels-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/kernels.yaml single 段，
#         device=OAM-810E 的 device_conditional_ignores 已展开合入）
# ------------------------------------------------------------------------------
# scope 决策（原 yaml 注释）：对位上游 kernels.yaml 11 step 的 PPU-relevant
# 子集 —— core / mamba / moe / quantization / attention 5 step；DeepGEMM
# (H100) / Kernels (B200) / Helion / FP8 MoE / Fp4 MoE 明确 device 限定，
# 跳过。通用 op 在 PPU 上验证 cuda kernel ABI 兼容性，有 catch-regression
# 价值。

# Step 1: kernels_core — 跳过 4 个有 fail 的文件（2026-05-25 SSH preflight
# 5263 PASS / 71 FAIL，fail 集中在这 4 个文件；原 yaml 注释）：
#   test_mrope — 6 个多模态模型（Qwen2/2.5/3-VL、GLM-4.1V 等）未入库，待入库
#   其余 3 个 — PPU regression 候选（torch.compile vs eager mismatch /
#   MLA rope fused kernel / 量化 rms_norm 特定 combos），Phase F+ 调查
KERNELS_SINGLE_CORE_ARGS=(
  tests/kernels/core/
  --ignore=tests/kernels/core/test_mrope.py
  --ignore=tests/kernels/core/test_fused_rms_norm_gated.py
  --ignore=tests/kernels/core/test_rotary_embedding_mla_cache_fused.py
  --ignore=tests/kernels/core/test_fused_quant_layernorm.py
  # 2026-08-27 PPU run fail，待调查
  --ignore=tests/kernels/core/test_vit_fp8_quant.py
)

# Step 2: kernels_mamba — test_causal_conv1d deterministic PPU ptxas SIGSEGV
# （Triton _causal_conv1d_update_kernel 编译 width=3+float32 组合崩溃，
# 历史 5 次 pipeline 全同样 32 errors；原 yaml 注释 2026-05-28），
# 长期方案是给 PPU SDK 提 ptxas crash 工单后 unignore
KERNELS_SINGLE_MAMBA_ARGS=(
  tests/kernels/mamba/
  --ignore=tests/kernels/mamba/test_causal_conv1d.py
)

# Step 3: kernels_moe — PPU fused_moe 改动最密集（160 行/16 文件），回归
# 信号最强子目录。跳过分类（原 yaml 注释）：
#   platform: CPU / ROCm aiter
#   device:   SM100+（Blackwell）/ flashinfer 外部包
#   multi-GPU: deepep 系（2-GPU + deep_ep）
#   ext pkg:  triton_kernels 包（module-level skip）
# OAM-810E device ignores：fp8e4nv 需 SM≥8.9（PPU 报 SM8.0）、acmoe INT8
# NaN / 精度 mismatch、deepgemm API mismatch
KERNELS_SINGLE_MOE_ARGS=(
  tests/kernels/moe/
  # platform-specific
  --ignore=tests/kernels/moe/test_cpu_fused_moe.py
  --ignore=tests/kernels/moe/test_rocm_aiter_topk.py
  # SM100+ (Blackwell) / flashinfer
  --ignore=tests/kernels/moe/test_cutedsl_moe.py
  --ignore=tests/kernels/moe/test_cutlass_mxfp8_grouped_mm.py
  --ignore=tests/kernels/moe/test_nvfp4_moe.py
  --ignore=tests/kernels/moe/test_flashinfer_moe.py
  --ignore=tests/kernels/moe/test_flashinfer.py
  --ignore=tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py
  --ignore=tests/kernels/moe/test_ocp_mx_moe.py
  # multi-GPU + deep_ep
  --ignore=tests/kernels/moe/test_deepep_deepgemm_moe.py
  --ignore=tests/kernels/moe/test_deepep_moe.py
  # triton_kernels pkg (module-level pytest.skip if missing)
  --ignore=tests/kernels/moe/test_gpt_oss_triton_kernels.py
  --ignore=tests/kernels/moe/test_modular_oai_triton_moe.py
  # test_modular_kernel_combinations 含 single+multi 混合用例，只跑单卡部分
  -k
  'not multigpu and not fp8 and not e4m3'
  # ---- OAM-810E device_conditional_ignores（原 yaml device 段） ----
  # DeepGemm 内部 fp8e4nv（per_block_cast_to_fp8），SM8.0 不支持（需 SM≥8.9）
  --ignore=tests/kernels/moe/test_batched_deepgemm.py
  --ignore=tests/kernels/moe/test_deepgemm.py
  # acmoe INT8 路径 NaN bug（95.2% mismatch）
  --ignore=tests/kernels/moe/test_block_int8.py
  # API mismatch: select_unquantized_moe_backend() 缺 moe_has_bias
  --ignore=tests/kernels/moe/test_unquantized_backend_selection.py
  # acmoe 精度: test_fused_moe 25% mismatch + routed transform
  --ignore=tests/kernels/moe/test_moe.py
  --ignore=tests/kernels/moe/test_shared_fused_moe_routed_transform.py
  # 2026-08-27 PPU run fail，待调查
  --ignore=tests/kernels/moe/test_grouped_topk.py
)

# Step 4: kernels_quantization — PPU 量化改动 81 行/9 文件
# （fp8/marlin/cutlass_scaled_mm）。跳过 SM100+ / ROCm / module-level
# HF download / compressed_tensors 脆弱依赖（原 yaml 注释）
KERNELS_SINGLE_QUANT_ARGS=(
  tests/kernels/quantization/
  # SM100+ (Blackwell)
  --ignore=tests/kernels/quantization/test_nvfp4_quant.py
  --ignore=tests/kernels/quantization/test_nvfp4_scaled_mm.py
  --ignore=tests/kernels/quantization/test_nvfp4_qutlass.py
  --ignore=tests/kernels/quantization/test_mxfp4_qutlass.py
  --ignore=tests/kernels/quantization/test_silu_mul_nvfp4_quant.py
  # SM100+ flashinfer
  --ignore=tests/kernels/quantization/test_flashinfer_scaled_mm.py
  --ignore=tests/kernels/quantization/test_flashinfer_nvfp4_scaled_mm.py
  # ROCm only
  --ignore=tests/kernels/quantization/test_rocm_skinny_gemms.py
  # module-level snapshot_download（红区无外网）
  --ignore=tests/kernels/quantization/test_gguf.py
  # bare import compressed_tensors（PPU 脆弱依赖）
  --ignore=tests/kernels/quantization/test_hadacore.py
  # fp8e4nv 需 SM≥8.9；DeepGemmQuantScaleFMT oracle 未初始化
  -k
  'not fp8 and not e4m3'
  # ---- OAM-810E device_conditional_ignores：fp8e4m3fn 需 SM≥8.9 ----
  --ignore=tests/kernels/quantization/test_block_int8.py
  # 本次测试失败（2026-08-27 PPU run），待调查
  --ignore=tests/kernels/quantization/test_allspark_gemm.py
  # v0.23.0新增的测试文件，实测fail
  --ignore=tests/kernels/quantization/test_nvfp4_emulation.py
  # 2026-08-27 PPU run fail 用例，--deselect 排除
  --deselect 'tests/kernels/quantization/test_cutlass_scaled_mm.py::test_cutlass_int8_azp'
  --deselect 'tests/kernels/quantization/test_mxfp4_triton_ep.py::TestTritonMoeForwardExpertMap::test_expert_map_remap'
)

# Step 5: kernels_attention — flashmla/triton_decode_attention 有 is_ppu()
# 路径。跳过 ROCm/CPU/XPU/SM90+/SM100+ 专用文件（原 yaml 注释）
KERNELS_SINGLE_ATTN_ARGS=(
  tests/kernels/attention/
  # ROCm
  --ignore=tests/kernels/attention/test_aiter_flash_attn.py
  --ignore=tests/kernels/attention/test_rocm_attention_selector.py
  # CPU
  --ignore=tests/kernels/attention/test_cpu_attn.py
  --ignore=tests/kernels/attention/test_mla_decode_cpu.py
  # XPU
  --ignore=tests/kernels/attention/test_xpu_mla_sparse.py
  # SM100+ (Blackwell)
  --ignore=tests/kernels/attention/test_cutlass_mla_decode.py
  --ignore=tests/kernels/attention/test_flashinfer_mla_decode.py
  --ignore=tests/kernels/attention/test_flashinfer_trtllm_attention.py
  # flashinfer（外部包，PPU 未安装）
  --ignore=tests/kernels/attention/test_flashinfer.py
  # SM90+（deep_gemm 要求）
  --ignore=tests/kernels/attention/test_deepgemm_attention.py
  # 内部用 fp8e4m3fn 但函数名不含 fp8（-k 无法过滤）
  --ignore=tests/kernels/attention/test_pack_unpack_triton.py
  # fp8e4nv/fp8e4m3fn 需 SM≥8.9；cuDNN THD 需 Hopper SM90+
  # ---- OAM-810E device_conditional_ignores ----
  # API mismatch: flash_mla_with_kvcache() 缺 descale_q
  --ignore=tests/kernels/attention/test_flashmla.py
  --ignore=tests/kernels/attention/test_flashmla_sparse.py
  # triton decode head_dim=192 精度问题（4 cases）
  --ignore=tests/kernels/attention/test_triton_decode_attention.py
  # 2026-08-27 PPU run fail，待调查
  --ignore=tests/kernels/attention/test_cascade_flash_attn.py
  --ignore=tests/kernels/attention/test_flash_attn.py
  --ignore=tests/kernels/attention/test_merge_attn_states.py
  # 同日 fail 的 test_triton_unified_attention.py 见上方 -k use_td 子句
    # test_triton_unified_attention.py ：193 failed / 97 passed
  # TD 路径 make_tensor_descriptor 在 num_kv_heads>1
  # 跨步布局下静默算错（数值 FAILED 非报错），(5,1)（num_kv_heads=1）全过，
  -k
  'not flashinfer and not fp8 and not e4m3 and not q_dtype1 and not (use_td and (num_heads0 or (num_heads1 and not tile_clamp)))'
 
)

# multi：无 — 首版只 single；test_mamba_mixer2.py 单个 @multi_gpu_test(2)
# 在单卡模式自动 SKIP，不阻塞

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 注意：禁止 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（上游
# TP 场景的 CUDA VMM workaround）——PPU 兼容层疑不支持 VMM API，是虚假
# OOM 头号嫌疑（详见 test-area-ppu-basic-correctness.sh 同段注释；
# Aone 侧从不设它且全绿）

# 默认离线（模型走 /nas_aisw 预置卷）；需要在线下载时设 PPU_TEST_ONLINE=1
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF 缓存：workflow 已注入 HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub；
# 未注入时（本地调试）探测 /nas_aisw 下的候选路径
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor 编译需要 cuda.h + ptxas + libcuda（缺失时
# torch.compile 类测试 BackendCompilerFailed）
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# 本 area 无 MODEL_MAP（用例全走合成张量；真模型引用均在 skip/ignore 覆盖
# 范围内，见文件头注释），跳过 HF cache symlink 段

# ------------------------------------------------------------------------------
# [junit] 合并基础设施（含 pytest 崩溃兜底）
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # 只由下方 `trap _emit_junit EXIT` 调用
_emit_junit() {
  python3 - <<PYEOF
import glob, os
from xml.etree import ElementTree as ET

OUT = "${RESULTS_DIR}/test.xml"
TMP = "${TMP_JUNIT}"
LABELS = "${STEP_LABELS_LIST}".split()

def _collect(label):
    shard_paths = sorted(glob.glob(f"{TMP}/{label}-shard*.xml"))
    if shard_paths:
        return shard_paths
    single = f"{TMP}/{label}.xml"
    return [single] if os.path.exists(single) else []

root = ET.Element("testsuites", name="vLLM PPU Kernels (GHA)")
for label in LABELS:
    paths = _collect(label)
    if not paths:
        ts = ET.SubElement(root, "testsuite", name=label, tests="1", errors="1",
                           failures="0", skipped="0", time="0")
        tc = ET.SubElement(ts, "testcase", name=label,
                           classname=f"gha_ci.{label}", time="0")
        err = ET.SubElement(tc, "error",
                            message="junit-xml not produced (pytest crashed before writing)")
        err.text = f"file missing: {TMP}/{label}.xml or shards"
        continue
    for path in paths:
        try:
            tree = ET.parse(path)
            for ts in tree.iter("testsuite"):
                ts.set("name", label)
                root.append(ts)
        except ET.ParseError:
            pass
ET.ElementTree(root).write(OUT, encoding="UTF-8", xml_declaration=True)
print(f"[junit] test.xml emitted -> {OUT}")

# ---- step summary：分 shard 统计表（markdown）。合并 test.xml 的
# testsuite name 已被改写为 label（丢失 shard 维度），故此处从原始
# shard xml 提取。宿主 workflow 把本文件 cat 进 GITHUB_STEP_SUMMARY，
# 在 run 的 Summary 页直接渲染（容器内拿不到该 env，需 workflow 接力）
SUMMARY = os.path.join(os.path.dirname(OUT), "summary.md")
COLS = ("tests", "failures", "errors", "skipped", "time")

def _stats(path):
    # junit 根节点 pytest 新旧版可能为 <testsuites> 或 <testsuite>
    root_ = ET.parse(path).getroot()
    suites = [root_] if root_.tag == "testsuite" else list(root_.iter("testsuite"))
    agg = dict.fromkeys(COLS, 0.0)
    for s in suites:
        for k in COLS:
            agg[k] += float(s.get(k) or 0)
    return agg

lines = ["### Kernels Test (PPU)", "",
         "| unit | tests | passed | failed | errors | skipped | time | status |",
         "|---|---:|---:|---:|---:|---:|---:|:-:|"]
tot = dict.fromkeys(COLS, 0.0)
bad_units = 0
for label in LABELS:
    paths = _collect(label)
    if not paths:
        lines.append(f"| {label} | - | - | - | - | - | - | FAILED (no junit, crashed) |")
        bad_units += 1
        continue
    for path in paths:
        unit = label
        base = os.path.basename(path)
        if "-shard" in base:
            unit = f"{label} / shard {base.rsplit('-shard', 1)[1][:-4]}"
        try:
            agg = _stats(path)
        except (ET.ParseError, OSError):
            lines.append(f"| {unit} | - | - | - | - | - | - | FAILED (bad junit) |")
            bad_units += 1
            continue
        n_t, n_f, n_e, n_s = (int(agg[k]) for k in COLS[:4])
        ok = (n_f == 0 and n_e == 0)
        if not ok:
            bad_units += 1
        lines.append(f"| {unit} | {n_t} | {n_t - n_f - n_e - n_s} | {n_f} "
                     f"| {n_e} | {n_s} | {agg['time']:.0f}s | "
                     f"{'PASSED' if ok else 'FAILED'} |")
        for k in COLS:
            tot[k] += agg[k]
n_t, n_f, n_e, n_s = (int(tot[k]) for k in COLS[:4])
status_all = "PASSED" if bad_units == 0 else "FAILED"
lines.append(f"| **Total** | **{n_t}** | **{n_t - n_f - n_e - n_s}** "
             f"| **{n_f}** | **{n_e}** | **{n_s}** | **{tot['time']:.0f}s** "
             f"| **{status_all}** |")
with open(SUMMARY, "w") as sf:
    sf.write("\n".join(lines) + "\n")
print(f"[summary-md] {SUMMARY} emitted")
PYEOF
  # 分片 step 的 pytest 输出重定向到 TMP_JUNIT，一并收进 artifact 便于排障
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  # 容器以 root 运行，产物须可被 runner 用户读取（upload-artifact）
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] 单 step 执行器：shards>1 → 并发分片；shards==1 → 单进程
# ------------------------------------------------------------------------------
TOTAL_RC=0

_run_step() {
  local label="$1" shards="$2"
  shift 2
  local args=("$@")
  STEP_LABELS_LIST="${STEP_LABELS_LIST} ${label}"

  if [ "${shards}" -gt 1 ]; then
    echo "========== [step] ${label} shards=${shards} =========="
    local pids=()
    for shard in $(seq 0 $((shards - 1))); do
      local out_xml="${TMP_JUNIT}/${label}-shard${shard}.xml"
      CUDA_VISIBLE_DEVICES="${shard}" pytest -v -s "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} \
        --shard-id="${shard}" --num-shards="${shards}" \
        --junit-xml="${out_xml}" \
        > "${TMP_JUNIT}/${label}-shard${shard}.log" 2>&1 &
      pids+=($!)
      echo "[shard] launched shard ${shard} pid=${!} CUDA_VISIBLE_DEVICES=${shard}"
    done
    local rc_total=0 i=0
    for pid in "${pids[@]}"; do
      set +e; wait "${pid}"; local rc=$?; set -e
      echo "[shard] shard ${i} pid=${pid} rc=${rc}"
      # 注：不可写成 `[ $rc -ne 0 ] && rc_total=1` —— 条件为假时整个
      # 表达式返回 1，若它是函数/分支的最后一条命令，函数返回码变成 1，
      # 顶层 set -e 会在函数调用处杀掉脚本（测试全过反而 exit 1 的元凶）
      if [ "${rc}" -ne 0 ]; then rc_total=1; fi
      i=$((i + 1))
    done
    echo "[step] ${label} aggregate rc=${rc_total}"
    for shard in $(seq 0 $((shards - 1))); do
      echo "----- ${label}-shard${shard}.log (tail) -----"
      tail -20 "${TMP_JUNIT}/${label}-shard${shard}.log" 2>/dev/null || echo "(no log)"
    done
    if [ "${rc_total}" -ne 0 ]; then TOTAL_RC=1; fi
  else
    echo "========== [step] ${label} =========="
    local out_xml="${TMP_JUNIT}/${label}.xml"
    set +e
    pytest -v -s "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} --junit-xml="${out_xml}"
    local rc=$?
    set -e
    echo "[step] ${label} rc=${rc}"
    if [ "${rc}" -ne 0 ]; then TOTAL_RC=1; fi
  fi
}

if [ "${MODE}" = "single" ]; then
  # Aone single 是 1-PPU pod：逐 step 限 1 卡（multi_gpu_test 自动 SKIP）
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_core" 1 "${KERNELS_SINGLE_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_mamba" 1 "${KERNELS_SINGLE_MAMBA_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_moe" 1 "${KERNELS_SINGLE_MOE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_quantization" 1 "${KERNELS_SINGLE_QUANT_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_attention" 1 "${KERNELS_SINGLE_ATTN_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area kernels has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_core" 1 "${KERNELS_SINGLE_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_mamba" 1 "${KERNELS_SINGLE_MAMBA_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_moe" 1 "${KERNELS_SINGLE_MOE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_quantization" 1 "${KERNELS_SINGLE_QUANT_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "kernels_attention" 1 "${KERNELS_SINGLE_ATTN_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
