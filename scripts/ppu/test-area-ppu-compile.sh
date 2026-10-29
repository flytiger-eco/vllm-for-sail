#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-compile.sh -- PPU Compile tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single | multi
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch pytest_args passthrough: split on whitespace and append to
# the tail of every pytest command (e.g. `-k test_foo -x`) so a rerun can be narrowed without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-compile-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the multi section in aone_ci/ppu_extras/compile.yaml)
# ------------------------------------------------------------------------------
# The compile area mirrors upstream .buildkite/test_areas/compile.yaml (13 steps) and only
# keeps the steps that exist and pass on PPU. They all need 2 cards (TP=2), so this area
# has no single cases, only the two multi steps:
#
#   OK   upstream Step 1-4 -> compile_correctness_e2e:
#        tests/compile/correctness_e2e/test_sequence_parallel.py (SP correctness) plus
#        tests/compile/correctness_e2e/test_async_tp.py (AsyncTP correctness).
#        Since run 34580922394, COMPILE_MULTI_E2E_ARGS drops the PPU-unsupported subset
#        (pp=1 pure SP / FP8 async_tp) with -k/--deselect; see the argument block.
#   OK   upstream Step 5 -> compile_passes_distributed:
#        tests/compile/passes/distributed/ (test_fusion_all_reduce / test_async_tp /
#        test_sequence_parallelism), which also covers upstream Step 6c (the standalone
#        test_fusion_all_reduce call). Since run 34580922394, COMPILE_MULTI_PASSES_ARGS
#        drops the FP8 / dtype1 ScaledMM / mnnvl+trtllm fusion failures; see that block.
#
# Per-step exclusion audit (snapshot of the ppu_extras header; keep the restore conditions):
#   SKIP upstream Step 6a: test_fusion_attn.py -k FLASHINFER - PPU has no FlashInfer, all skip
#   SKIP upstream Step 6b: test_silu_mul_quant_fusion.py - all fail
#        (Expected a.dtype()==kInt8, the cutlass_scaled_mm PPU kernel does not support it)
#   SKIP upstream Step 6d: test_full_graph.py::test_fp8_kv_scale_compile - all fail
#        (model LocalEntryNotFoundError + FlashInfer MLA compute capability unsupported)
#   SKIP upstream Step 7-9: fusions_e2e/test_tp1_quant.py - 32 failed / 264 skipped
#   SKIP upstream Step 10-13: fusions_e2e/test_tp2_ar_rms.py / test_tp2_async_tp.py
#        - the same TP1 problems, excluded
#   Restore: enable the upstream steps one by one once PPU supports FlashInfer / the
#   cutlass_scaled_mm int8 kernel / MLA compute capability.
#
# Every upstream step exports `VLLM_TEST_CLEAN_GPU_MEMORY=1`; the Aone sh does not
# reproduce that env and this script follows Aone (parity with the all-green reference).
COMPILE_MULTI_E2E_ARGS=(
  tests/compile/correctness_e2e/test_sequence_parallel.py
  tests/compile/correctness_e2e/test_async_tp.py
# Run 34580922394: the pp=1 pure SP combinations (parallel_setup0-3/8-11 x eager x
# inductor, 16 cases) plus prompt_embeds parallel_setup0 (2 cases) raise "Server exited
# unexpectedly" on 2 PPU cards. pp=2 (setup4-7/12-15) and FP8 SP (setup16-19) pass and
# stay. The trailing dash in parallel_setup1 avoids the passing setup12-19 and
# prompt_embeds "parallel_setup1]" cases; parallel_setup0 has no trailing dash so it also
# covers the failing prompt_embeds "setup0]" id.
# Restore: drop this -k once the PPU pp=1 SP crash is fixed.
  -k
  "not parallel_setup0 and not parallel_setup1- and not parallel_setup2 and not parallel_setup3 and not parallel_setup8 and not parallel_setup9 and not parallel_setup10 and not parallel_setup11"
# Run 34580922394: the 2 Meta-Llama-3.1-8B-Instruct-FP8 async_tp cases fail (fp8e4nv needs
# SM8.9+ and PPU OAM-810E reports SM8.0); the 2 base-1B cases pass and stay.
# Restore: drop them once PPU supports fp8e4nv.
  --deselect "tests/compile/correctness_e2e/test_async_tp.py::test_async_tp_pass_correctness[False-mp-True-2-RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8]"
  --deselect "tests/compile/correctness_e2e/test_async_tp.py::test_async_tp_pass_correctness[True-mp-True-2-RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8]"
# Run 34936679139: the 1B False combination raises RuntimeError: Server failed to start
# in time (startup timeout; the same case passed in run 34580922394, so it looks like a
# timing flake). The True combination still passes and stays. Restore once startup is stable.
  --deselect "tests/compile/correctness_e2e/test_async_tp.py::test_async_tp_pass_correctness[False-mp-True-2-meta-llama/Llama-3.2-1B-Instruct]"
)

COMPILE_MULTI_PASSES_ARGS=(
  tests/compile/passes/distributed/
# Run 34580922394 (the 56 failed / 38 passed design):
# 1) all 32 TestAllReduceRMSNormStaticQuantFP8Model cases in test_sequence_parallelism.py
#    fail (fp8e4nv needs SM8.9+); the other model classes in that file pass, so exclude by
#    class name, which also covers the 8 mnnvl/trtllm failures of the same class in
#    test_fusion_all_reduce.py.
# 2) of the 4 ScaledMM models in test_async_tp.py::test_async_tp_pass_replace only the 8
#    dtype1 cases fail (cutlass_scaled_mm_sm80_epilogue scaled_mm_c2x.cu:89, PPU lacks
#    that kernel); the dtype0 ScaledMM cases pass, so the filter must keep dtype1.
# Restore: once PPU supports fp8e4nv / the cutlass_scaled_mm int8 kernel.
  -k
  "not TestAllReduceRMSNormStaticQuantFP8Model and not (dtype1 and ScaledMM)"
# Run 34580922394 (continued): 8 GemmaRMSNorm/RMSNorm cases in test_fusion_all_reduce.py
# report matched_count=0 for the mnnvl/trtllm backends (PPU has neither, so the fusion
# pass never matches); the 4 FP4 cases in that file pass and the 8 FP8 ones are already
# excluded above, so these are deselected per case instead of ignoring the whole file.
# Restore: once PPU supports the mnnvl/trtllm allreduce backends.
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[mnnvl-False-dtype0-64-8-8-TestAllReduceGemmaRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[mnnvl-False-dtype0-64-8-8-TestAllReduceRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[mnnvl-True-dtype0-64-8-8-TestAllReduceGemmaRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[mnnvl-True-dtype0-64-8-8-TestAllReduceRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[trtllm-False-dtype0-64-8-8-TestAllReduceGemmaRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[trtllm-False-dtype0-64-8-8-TestAllReduceRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[trtllm-True-dtype0-64-8-8-TestAllReduceGemmaRMSNormModel-False-False]"
  --deselect "tests/compile/passes/distributed/test_fusion_all_reduce.py::test_all_reduce_fusion_pass_replace[trtllm-True-dtype0-64-8-8-TestAllReduceRMSNormModel-False-False]"
)

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
# torch.compile tests fail with BackendCompilerFailed) - this area is mostly torch.compile
# / inductor paths, so this block is essential
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
# (absolute NAS path = /nas_aisw/datasets/ + path); entries missing from the list are
# looked up in aone_ci/scripts/ppu_model_aliases.json (/ppusw/ replaced by /nas_aisw/).
# A missing path warns and is skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
# test_sequence_parallel.py (SPTestSettings.fast) + the tiny base of test_async_tp.py
# list MISS; the aone ppu_model_aliases.json (tiny/v1.0) hits and the bc blueprint already
# confirmed it with ls on the runner (tiny/v1.0/ holds 4 tiny models absent from cleaned.json)
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
# test_sequence_parallel.py (SPTestSettings.fp8_quant) + the FP8 base of test_async_tp.py
# list hit: checkpoints_cleaned.json ms_name=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
    # path=checkpoints/LLM/Llama/v3.1/Meta-Llama-3.1-8B-Instruct-FP8
    "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.1/Meta-Llama-3.1-8B-Instruct-FP8",
# the non-quantized base of test_async_tp.py
# list hit: checkpoints_cleaned.json name=Llama-3.2-1B-Instruct
# path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct (the Aone /ppusw side has a v3.1
# directory and the red-zone list's v3.2 wins, same as the bc blueprint)
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
# the FP8 model shared by the three files in tests/compile/passes/distributed/ (test_async_tp /
# test_fusion_all_reduce / test_sequence_parallelism)
# list MISS; hits aone ppu_model_aliases.json line 201
    # /ppusw/datasets/checkpoints/LLM/Llama/v1.0/Llama-3.2-1B-Instruct-FP8
    "RedHatAI/Llama-3.2-1B-Instruct-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-3.2-1B-Instruct-FP8",
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

root = ET.Element("testsuites", name="vLLM PPU Compile (GHA)")
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

lines = ["### Compile Test (PPU)", "",
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

# The compile area has no single cases (matching aone_ci/scripts/test_area_ppu_compile.sh:
# single mode -> "no single steps configured" -> exit 0); the two multi steps each run as
# one process and the cases take 2 cards through TP=2 (no CUDA_VISIBLE_DEVICES limit, so
# every card stays visible).
if [ "${MODE}" = "single" ]; then
  echo "[run] no single steps configured for compile area"
elif [ "${MODE}" = "multi" ]; then
  _run_step "compile_correctness_e2e" 1 "${COMPILE_MULTI_E2E_ARGS[@]}"
  _run_step "compile_passes_distributed" 1 "${COMPILE_MULTI_PASSES_ARGS[@]}"
else  # all
  _run_step "compile_correctness_e2e" 1 "${COMPILE_MULTI_E2E_ARGS[@]}"
  _run_step "compile_passes_distributed" 1 "${COMPILE_MULTI_PASSES_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
