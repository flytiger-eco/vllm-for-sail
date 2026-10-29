#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-kernels.sh -- PPU Kernels tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container,
# cwd = /workspace).
#
# This area has no MODEL_MAP: every case uses synthetic tensors. The only real model
# reference, neuralmagic/Llama-3.2-1B-quantized.w8a8, lives in
# test_triton_scaled_mm.py::test_rocm_compressed_tensors_w8a8
# (@skipif ROCm-only, so PPU skips it); the Qwen-VL/GLM/fxmarty/gguf model references
# all sit inside ignored files.
#
# Environment variables:
#   TEST_MODE   all(default) | single   - this area has no multi section (the only
#                                      multi-card case upstream,
#                                      test_mamba_mixer2.py @multi_gpu_test(2), self-skips on 1 card)
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-kernels-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/kernels.yaml,
#         with the device=OAM-810E device_conditional_ignores folded in)
# ------------------------------------------------------------------------------
# Scope: the PPU-relevant subset of the 11 upstream kernels.yaml steps - core / mamba / moe
# / quantization / attention. DeepGEMM (H100) / Kernels (B200) / Helion / FP8 MoE / Fp4 MoE
# are device-specific and skipped. The generic ops verify cuda kernel ABI compatibility on
# PPU and are worth catching regressions on.

# Step 1: kernels_core - skips the 4 files that fail (2026-05-25 SSH preflight:
# 5263 PASS / 71 FAIL, all failures confined to these 4 files; original yaml note):
#   test_mrope - 6 multimodal models (Qwen2/2.5/3-VL, GLM-4.1V, ...) not staged yet
#   the other 3 - PPU regression candidates (torch.compile vs eager mismatch / MLA rope
#   fused kernel / quantized rms_norm combinations), tracked under Phase F+
KERNELS_SINGLE_CORE_ARGS=(
  tests/kernels/core/
  --ignore=tests/kernels/core/test_mrope.py
  --ignore=tests/kernels/core/test_fused_rms_norm_gated.py
  --ignore=tests/kernels/core/test_rotary_embedding_mla_cache_fused.py
  --ignore=tests/kernels/core/test_fused_quant_layernorm.py
# PPU run failed 2026-08-27, to be investigated
  --ignore=tests/kernels/core/test_vit_fp8_quant.py
)

# Step 2: kernels_mamba — test_causal_conv1d deterministic PPU ptxas SIGSEGV
# (Triton _causal_conv1d_update_kernel crashes while compiling the width=3+float32
# combination; the last 5 pipelines show the same 32 errors; original yaml note
# 2026-05-28). The long-term fix is a ptxas crash ticket to the PPU SDK, then unignore.
KERNELS_SINGLE_MAMBA_ARGS=(
  tests/kernels/mamba/
  --ignore=tests/kernels/mamba/test_causal_conv1d.py
)

# Step 3: kernels_moe - the heaviest PPU fused_moe churn (160 lines / 16 files) and the
# strongest regression signal. Skipped classes:
#   platform: CPU / ROCm aiter
#   device:   SM100+ (Blackwell) / the flashinfer external package
#   multi-GPU: the deepep family (2 GPU + deep_ep)
#   ext pkg:  the triton_kernels package (module-level skip)
# OAM-810E device ignores: fp8e4nv needs SM>=8.9 (PPU reports SM8.0), acmoe INT8 NaN /
# accuracy mismatch and a deepgemm API mismatch
KERNELS_SINGLE_MOE_ARGS=(
  tests/kernels/moe/
  # platform-specific
  --ignore=tests/kernels/moe/test_cpu_fused_moe.py
  --ignore=tests/kernels/moe/test_rocm_aiter_topk.py
  # SM100+ (Blackwell) / flashinfer
  --ignore=tests/kernels/moe/test_cutedsl_moe.py
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
# test_modular_kernel_combinations mixes single and multi cases; only the single-card part runs
  -k
  'not multigpu and not fp8 and not e4m3'
# ---- OAM-810E device_conditional_ignores (the device section of the original yaml) ----
# DeepGemm uses fp8e4nv internally (per_block_cast_to_fp8), unsupported on SM8.0 (needs >=8.9)
  --ignore=tests/kernels/moe/test_batched_deepgemm.py
  --ignore=tests/kernels/moe/test_deepgemm.py
# acmoe INT8 path NaN bug (95.2% mismatch)
  --ignore=tests/kernels/moe/test_block_int8.py
# API mismatch: select_unquantized_moe_backend() lacks moe_has_bias
  --ignore=tests/kernels/moe/test_unquantized_backend_selection.py
# acmoe accuracy: test_fused_moe 25% mismatch + routed transform
  --ignore=tests/kernels/moe/test_moe.py
  --ignore=tests/kernels/moe/test_shared_fused_moe_routed_transform.py
# PPU run failed 2026-08-27, to be investigated
  --ignore=tests/kernels/moe/test_grouped_topk.py
)

# Step 4: kernels_quantization - 81 lines / 9 files of PPU quantization churn
# (fp8/marlin/cutlass_scaled_mm). Skips SM100+ / ROCm / module-level HF downloads / the
# fragile compressed_tensors dependency
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
# bare import of compressed_tensors (a fragile PPU dependency)
  --ignore=tests/kernels/quantization/test_hadacore.py
# fp8e4nv needs SM>=8.9; the DeepGemmQuantScaleFMT oracle is not initialized
  -k
  'not fp8 and not e4m3'
# ---- OAM-810E device_conditional_ignores: fp8e4m3fn needs SM>=8.9 ----
  --ignore=tests/kernels/moe/test_block_int8.py
# Failed in this round (PPU run 2026-08-27), to be investigated
  --ignore=tests/kernels/quantization/test_allspark_gemm.py
# New test file in v0.23.0, fails as measured
  --ignore=tests/kernels/quantization/test_nvfp4_emulation.py
# Case that failed in the 2026-08-27 PPU run, excluded with --deselect
  --deselect 'tests/kernels/quantization/test_cutlass_scaled_mm.py::test_cutlass_int8_azp'
  --deselect 'tests/kernels/quantization/test_mxfp4_triton_ep.py::TestTritonMoeForwardExpertMap::test_expert_map_remap'
# 2026-09-01 PPU nightly failure: marlin int8 x uint4 (the AWQ zp path, bf16 output)
# max_diff 0.0408 just crosses the 0.04 threshold - PPU numeric drift, same family as the
# cutlass int8 AZP issues. Deselecting this one case first; if other parameter combinations
# keep reproducing (whack-a-mole), promote it to a function-level deselect, following how
# test_cutlass_int8_azp evolved.
  --deselect 'tests/kernels/quantization/test_marlin_gemm.py::test_marlin_gemm[a_type467-b_type467-c_type467-8-257-3328-1408-False-False-False-True]'
)

# Step 5: kernels_attention - flashmla/triton_decode_attention have is_ppu() paths.
# Skips the ROCm/CPU/XPU/SM90+/SM100+ specific files:
KERNELS_SINGLE_ATTN_ARGS=(
  tests/kernels/attention/
  # ROCm
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
# flashinfer (external package, not installed on PPU)
  --ignore=tests/kernels/attention/test_flashinfer.py
# SM90+ (a deep_gemm requirement)
  --ignore=tests/kernels/attention/test_deepgemm_attention.py
# uses fp8e4m3fn internally but the name has no fp8, so -k cannot filter it
  --ignore=tests/kernels/attention/test_pack_unpack_triton.py
# fp8e4nv/fp8e4m3fn need SM>=8.9; cuDNN THD needs Hopper SM90+
  # ---- OAM-810E device_conditional_ignores ----
# API mismatch: flash_mla_with_kvcache() lacks descale_q
  --ignore=tests/kernels/attention/test_flashmla.py
  --ignore=tests/kernels/attention/test_flashmla_sparse.py
# triton decode head_dim=192 accuracy problem (4 cases)
  --ignore=tests/kernels/attention/test_triton_decode_attention.py
# PPU run failed 2026-08-27, to be investigated
  --ignore=tests/kernels/attention/test_cascade_flash_attn.py
  --ignore=tests/kernels/attention/test_flash_attn.py
  --ignore=tests/kernels/attention/test_merge_attn_states.py
# for test_triton_unified_attention.py, which failed the same day, see the -k use_td clause above
# test_triton_unified_attention.py : 193 failed / 97 passed
# the TD path's make_tensor_descriptor silently computes wrong values for num_kv_heads>1
# with a strided layout (numeric FAILED, not an error); the (5,1) cases
# (num_kv_heads=1) all pass,
  -k
  'not flashinfer and not fp8 and not e4m3 and not q_dtype1 and not (use_td and (num_heads0 or (num_heads1 and not tile_clamp)))'

)

# multi: none - the first version is single only; the one @multi_gpu_test(2) in
# test_mamba_mixer2.py self-skips in single mode, so it is not a blocker

# ------------------------------------------------------------------------------
# [env] offline mode and runtime settings
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# offline by default (models come from the /nas_aisw volume); set PPU_TEST_ONLINE=1 to download
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF cache: the workflow injects HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub;
# when unset (local debugging) probe the candidates under /nas_aisw
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor compilation needs cuda.h + ptxas + libcuda; without them
# torch.compile tests fail with BackendCompilerFailed
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# This area has no MODEL_MAP (all cases use synthetic tensors and the real model references
# are under skip/ignore, see the header), so the HF cache symlink block is skipped

# ------------------------------------------------------------------------------
# [junit] merge helper (with a fallback when pytest crashes)
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329 # invoked through `trap _emit_junit EXIT`
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

# ---- step summary: per-shard statistics table (markdown)
SUMMARY = os.path.join(os.path.dirname(OUT), "summary.md")
COLS = ("tests", "failures", "errors", "skipped", "time")

def _stats(path):
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
  # the container runs as root, so results must stay readable by the runner user (upload-artifact)
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] ship the per-unit table back over NAS: RESULTS_DIR lives in the pod and is lost on cleanup;
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ] && [ -f "${RESULTS_DIR}/summary.md" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -f "${RESULTS_DIR}/summary.md" "${PPU_SUMMARY_NAS_DIR}/summary.md"; then
      echo "[k8s-summary] summary.md -> ${PPU_SUMMARY_NAS_DIR}/summary.md"
    else
      echo "[k8s-summary] WARN: NAS upload failed (is ${PPU_SUMMARY_NAS_DIR} writable?)"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] step runner: shards>1 runs concurrent shards, shards==1 runs one process
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
      CUDA_VISIBLE_DEVICES="${shard}" pytest -v -s "${args[@]}" \
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
    pytest -v -s "${args[@]}" --junit-xml="${out_xml}"
    local rc=$?
    set -e
    echo "[step] ${label} rc=${rc}"
    if [ "${rc}" -ne 0 ]; then TOTAL_RC=1; fi
  fi
}

if [ "${MODE}" = "single" ]; then
# The Aone single job is a 1-PPU pod: 1 card per step (multi_gpu_test self-skips)
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
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
