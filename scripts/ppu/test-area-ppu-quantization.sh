#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-quantization.sh -- PPU Quantization tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single   - this area is single-only (no multi step)
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac
# quantization is single-only; the multi mode has no steps to run (see the dispatch at the bottom)

# workflow_dispatch pytest_args passthrough: split on whitespace and append to
# the tail of every pytest command (e.g. `-k test_foo -x`) so a rerun can be narrowed without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-quantization-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area-specific dependencies: none.
# ------------------------------------------------------------------------------
# The upstream .buildkite/test_areas/quantization.yaml "Quantization" step installs
# torchao==0.14.1 + conch-triton-kernels, but the PPU side ignores test_torchao.py entirely
# (torchao needs SM>=8.9 and the version is too old) and the conch cases do not run either, so
# nothing to add. bitsandbytes / auto-round / AWQ come from the image stack (the ppu_extras
# audit confirmed bitsandbytes is installed). If the first run finds a gap, install it here
# after an import probe.

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/quantization.yaml)
# ------------------------------------------------------------------------------
# Mapping of the 3 upstream steps:
#   Step 1 "Quantization"              -> tests/quantization/ (17 files, per-file audit below)
#   Step 2 "Quantized MoE Test (B200)" -> whole step skipped (SM100+ Blackwell, unsupported on PPU)
#   Step 3 "Quantized Models Test"     -> tests/models/quantization/ (9 files)
#
# quantization_core = the PPU version of upstream Step 1: all of tests/quantization/ minus 14
#   ignores, the fp8 kv -k filter and 10 deselects. After run 34580922394 it actually runs:
#   the PPU-capable subset of test_auto_round / test_configs / test_experts_int8 (the root
#   cause and restore condition of each exclusion are noted inside the argument block).
QUANT_CORE_ARGS=(
  tests/quantization/
# SM100+ Blackwell, needs flashinfer
  --ignore=tests/quantization/test_blackwell_moe.py
# CPU-only (module-level pytest.skip if not CPU)
  --ignore=tests/quantization/test_cpu_wna16.py
# fp8 core, needs SM>=8.9 (PPU reports SM8.0)
# (the upstream ROCm-only test_ptpc_fp8.py merged into test_quark.py, so the dead ignore is gone)
  --ignore=tests/quantization/test_fp8.py
# fp8 dependency (SM>=8.9) plus many HF models
  --ignore=tests/quantization/test_cpu_offload.py
# fp8 (SM>=8.9) + HF snapshot_download
  --ignore=tests/quantization/test_modelopt.py
  # AMD quark + lm_eval + HF access
  --ignore=tests/quantization/test_quark.py
  --ignore=tests/quantization/test_mixed_precision.py
# torchao is installed but too old (needs >=0.14.0) and fp8 Float8 needs CUDA>=8.9
  --ignore=tests/quantization/test_torchao.py
# the compressed_tensors package is not in the PPU CI image and the file holds many fp8/SM90+ cases
  --ignore=tests/quantization/test_compressed_tensors.py
# model not staged in the red zone (no internet)
  --ignore=tests/quantization/test_lm_head.py
  --ignore=tests/quantization/test_gptq_dynamic.py
  --ignore=tests/quantization/test_gptq_v2.py
# Run 34580922394: 6/6 fail - fp8e4nv needs SM8.9+ (PPU reports SM8.0) and
# is_quant_method_supported("fp8") wrongly reports supported on PPU, so nothing is skipped.
  --ignore=tests/quantization/test_online.py
# Run 34580922394: all 23 fp8 parametrizations of test_per_token_kv_cache.py fail
# (Triton: type fp8e4nv not supported, PPU only supports fp8e4b15/fp8e5);
# the "[fp8-"/"[fp8]" prefixes match the failing ids exactly and keep passing cases such as
# [fp8_per_token_head] (no kernel path) plus all int8 cases.
  -k
  "not [fp8- and not [fp8]"
# Run 34580922394: TheBloke/Llama-2-7B-Chat-GPTQ and
# TheBloke/OpenHermes-2.5-Mistral-7B-AWQ are not staged ([setup] MISS), so the exptype0/1/2/8/9/10
# variants that expect successful detection fail (the variants expecting ERROR still pass).
# Restore: drop these 6 deselects once the two checkpoints are staged.
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype0]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype1]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype2]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype8]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype9]"
  --deselect "tests/quantization/test_configs.py::test_auto_gptq[model_arg_exptype10]"
# Run 34580922394: TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ is not staged (no such NAS path),
# LocalEntryNotFoundError. Restore: drop once the checkpoint is staged.
  --deselect "tests/quantization/test_auto_gptq.py::test_auto_gptq_quantization_method[TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ]"
# Run 34580922394: the Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound engine fails to initialise
# (PPU torch_call_dispatcher aten::sum dim_IntList API call failed); the OPEA/Qwen2.5 variant in
# the same file passes. Restore: drop once PPU aten::sum is fixed.
  --deselect "tests/quantization/test_auto_round.py::test_auto_round[Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound]"
# Run 34580922394: the Jamba-tiny-random engine raises a ModelConfig ValidationError at startup
# (vllm rejects the staged red-zone snapshot content); the plamo variant skips by itself.
  --deselect "tests/quantization/test_experts_int8.py::test_model_experts_int8_startup[4-bfloat16-ai21labs/Jamba-tiny-random]"
# Run 34580922394: PPU torch.linalg.det numeric drift (|det|-1 = 0.1665, tol=1e-4),
# single case excluded.
  --deselect "tests/quantization/test_turboquant.py::TestRotationMatrix::test_rotation_matrix_det_is_pm1"
)

# quantization_models = the PPU version of upstream Step 3: tests/models/quantization/ minus 10
#   ignores and the fp8 kv -k filter. After run 34580922394 it actually runs:
#   the int8 cases of test_per_token_kv_cache.py (the fp8 variants are dropped by -k).
QUANT_MODELS_ARGS=(
  tests/models/quantization/
# SM100+ (Blackwell)
  --ignore=tests/models/quantization/test_nvfp4.py
  # AMD quark MXFP4
  --ignore=tests/models/quantization/test_mxfp4.py
# fp8 (SM>=8.9)
  --ignore=tests/models/quantization/test_fp8.py
  --ignore=tests/models/quantization/test_modelopt.py
# AMD gpt_oss (quark + lm_eval)
  --ignore=tests/models/quantization/test_gpt_oss.py
# model not staged (TechxGenus/gemma-1.1-2b-it-GPTQ; the parametrization is per file, so
# model1 TheBloke/TinyLlama passes while model2 gemma fails -> whole file ignored)
  --ignore=tests/models/quantization/test_gptq_marlin.py
# Run 34580922394: 5/5 fail - the gemma4-moe AWQ model is not staged
# (LocalEntryNotFoundError) plus InternVL2-2B EngineDeadError (c10::Error).
# Restore: unignore once gemma4-moe AWQ is staged and InternVL2 is fixed on PPU.
  --ignore=tests/models/quantization/test_awq.py
# Run 34580922394: 4/4 fail - ValueError: Failed to find a kernel that can
# implement the MXFP8 linear layer (PPU has no MXFP8 kernel implementation).
  --ignore=tests/models/quantization/test_mxfp8.py
# Run 34580922394: 1 fp8_per_token_head case fails (fp8e4nv needs SM8.9+, PPU reports
# SM8.0); the int8 cases in the same file pass, hence the precise -k instead of a file ignore.
  -k
  "not fp8_per_token_head"
)

# v0.28 moved the bitsandbytes tests into plugins_tests/bitsandbytes, owned by the plugins area;
# the quantization area no longer collects them.

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

# ------------------------------------------------------------------------------
# [setup] HF hub cache symlink: HF id -> local checkpoint under /nas_aisw
# ------------------------------------------------------------------------------
# paths come from the red-zone model list's path field
# (absolute NAS path = /nas_aisw/datasets/ + path); entries missing from the list follow the Aone
# /ppusw-isomorphic path and are marked "unconfirmed". A missing path warns and is skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # ---- quantization_core: tests/quantization/ ----
# test_auto_round.py (AutoRound INT4)
# main list hit name=Qwen2-0.5B-Instruct-int4-sym-AutoRound
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/"
        "Qwen2-0.5B-Instruct-int4-sym-AutoRound",
# main list hit name=Qwen2.5-0.5B-Instruct-int4-sym-inc
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/"
        "Qwen2.5-0.5B-Instruct-int4-sym-inc",
# test_configs.py (GPTQ/AWQ config parsing)
# main list MISS, Aone alias hit (/ppusw->/nas_aisw)
    "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/"
        "TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
    "TheBloke/Llama-2-7B-Chat-GPTQ":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-2-7B-Chat-GPTQ",
    "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ":
        "/nas_aisw/datasets/checkpoints/LLM/OpenHermes/v1.0/"
        "OpenHermes-2.5-Mistral-7B-AWQ",
# test_experts_int8.py (INT8 experts startup)
# main list MISS, Aone alias hit
    "ai21labs/Jamba-tiny-random":
        "/nas_aisw/datasets/checkpoints/LLM/Jamba/v1.0/Jamba-tiny-random",
    "pfnet/plamo-2-1b":
        "/nas_aisw/datasets/checkpoints/LLM/plamo/v1.0/plamo-2-1b",
# test_register_quantization_config.py (custom quantization config + a small LLM)
# main list hit path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct
# (the list ms_name=unsloth/..., matched by name; the Aone /ppusw side has v3.1, so the
# main list v3.2 wins)
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",

}

HF_CACHE = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
    "~/.cache/huggingface/hub")
os.makedirs(HF_CACHE, exist_ok=True)

created = skipped = missing = 0
for hf_id, local_path in MODEL_MAP.items():
    if not os.path.isdir(local_path):
        print(f"[setup] MISS  {hf_id} -> {local_path} (tests using it will fail)")
        missing += 1
        continue
    org, repo = hf_id.split("/", 1)
    cache_dir = os.path.join(HF_CACHE, f"models--{org}--{repo}")
    snap_link = os.path.join(cache_dir, "snapshots", "main")
    refs_file = os.path.join(cache_dir, "refs", "main")
    if os.path.islink(snap_link) or os.path.exists(snap_link):
        skipped += 1
        continue
    os.makedirs(os.path.dirname(snap_link), exist_ok=True)
    os.makedirs(os.path.dirname(refs_file), exist_ok=True)
    os.symlink(local_path, snap_link)
    with open(refs_file, "w") as rf:
        rf.write("main")
    created += 1
    print(f"[setup] OK    {hf_id} -> {local_path}")
print(f"[setup] symlinks: created={created} skipped={skipped} "
      f"missing={missing} (total {len(MODEL_MAP)})")
PYEOF

# ------------------------------------------------------------------------------
# [junit] merge helper (with a fallback when pytest crashes)
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # only invoked by `trap _emit_junit EXIT` below
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

root = ET.Element("testsuites", name="vLLM PPU Quantization")
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

lines = ["### Quantization Test (PPU)", "",
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
  # sharded steps send pytest output to TMP_JUNIT, shipped in the artifact for triage
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
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
# quantization is single-only: 2 steps with 1 card each (no multi_gpu_test cases)
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_core" 1 "${QUANT_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_models" 1 "${QUANT_MODELS_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area quantization has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_core" 1 "${QUANT_CORE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "quantization_models" 1 "${QUANT_MODELS_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
