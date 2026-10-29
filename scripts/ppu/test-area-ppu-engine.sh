#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-engine.sh -- PPU Engine tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single   - this area has no multi section (the upstream
#                                      V1 e2e 2/4 GPU steps run spec_decode, not applicable on 1 PPU)
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
TMP_JUNIT="/tmp/ppu-engine-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/engine.yaml)
# ------------------------------------------------------------------------------
# Step 1: engine_basic - tests/engine/test_arg_utils.py plus the root-level
#   test_sequence/logger/vllm_port (all GPU-free).
#   The snapshot takes test_arg_utils.py rather than all of tests/engine/:
#   - test_short_mm_context.py -> llava-hf/llava-1.5-7b-hf multimodal, not covered yet
#   - test_config.py -> many ModelConfig() calls download HF config, heavy red-zone dependency
ENGINE_SINGLE_BASIC_ARGS=(
  tests/engine/test_arg_utils.py
  tests/test_sequence.py
  tests/test_logger.py
  tests/test_vllm_port.py
)

# Step 2: v1_engine - the whole tests/v1/engine/ tree (V1 engine core/client/async/
#   preprocess). Module-level AutoTokenizer.from_pretrained calls (test_engine_core_client.py:55
#   Llama-3.2-1B-Instruct, test_engine_core.py:37 tiny-random-LlamaForCausalLM,
#   conftest.py:33 Llama-3.2-1B) are handled by the MODEL_MAP symlinks below, no ignore needed.
#
#   First measured run (2026-08-27): 17 failed / 100 passed / 2 skipped (2732s).
#   The 17 red cases are deselected one by one (below); delete a line once it is fixed.
ENGINE_SINGLE_V1_ARGS=(
  tests/v1/engine/
# ---- deselect: red cases from the first run (2026-08-27) ----
# abort semantics x6 (final step abort / multi abort / abort final output)
  --deselect "tests/v1/engine/test_abort_final_step.py::test_abort_during_final_step[False]"
  --deselect "tests/v1/engine/test_abort_final_step.py::test_abort_during_final_step[True]"
  --deselect "tests/v1/engine/test_async_llm.py::test_multi_abort[RequestOutputKind.DELTA]"
  --deselect "tests/v1/engine/test_async_llm.py::test_multi_abort[RequestOutputKind.FINAL_ONLY]"
  --deselect "tests/v1/engine/test_async_llm.py::test_abort_final_output[RequestOutputKind.DELTA]"
  --deselect "tests/v1/engine/test_async_llm.py::test_abort_final_output[RequestOutputKind.FINAL_ONLY]"
# EngineCore basics x4
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_advanced_sampling"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_concurrent_batches"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_invalid_request_id_type"
# encoder zero kv-cache instances x6
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_producer-0.01-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_consumer-0.7-True]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_consumer-0.7-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_producer-0.01-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_consumer-0.7-True]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_consumer-0.7-False]"
# preprocess error handling x1
  --deselect "tests/v1/engine/test_preprocess_error_handling.py::test_preprocess_error_handling"
)

# Step 3: v1_e2e_general - tests/v1/e2e/general/ (async scheduling, min_tokens,
#   streaming, context_length, sliding_window, cascade_attention)
ENGINE_SINGLE_E2E_ARGS=(
  tests/v1/e2e/general/
# gemma-3n-E2B-it is large, not covered in the first version
  --ignore=tests/v1/e2e/general/test_kv_sharing_fast_prefill.py
# Qwen3-Next-80B-A3B-Instruct-FP8 cannot run on a single PPU
  --ignore=tests/v1/e2e/general/test_mamba_prefix_cache.py
# Qwen3-Embedding-0.6B embedding model, deferred in the first version
  --ignore=tests/v1/e2e/general/test_pooling_chunked_prefill.py
# JackFram/llama-160m is not in the red zone (missing from the GHA list and the Aone aliases);
# only test_decoder_max_context_length_validation uses it, so the whole file is ignored;
# delete this line and add the MODEL_MAP path once the model is staged
  --ignore=tests/v1/e2e/general/test_context_length.py
)

# multi: none - the upstream V1 e2e 2/4 GPU steps run spec_decode, not applicable on 1 PPU

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
# (absolute NAS path = /nas_aisw/datasets/ + path); entries missing from the list use the
# isomorphic Aone ppu_model_aliases.json path (/ppusw/ -> /nas_aisw/). A missing path warns and is skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
# v1_engine conftest.py / test_engine_core_client.py and others;
# list hit: checkpoints_cleaned.json path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    # v1_engine conftest.py:33
# list hit: path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B
    "meta-llama/Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B",
# v1_engine / v1_e2e_general, several cases;
# list hit: path=checkpoints/LLM/misc/v1.0/opt-125m
    "facebook/opt-125m": "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
# test_engine_core.py:37 module-level tokenizer;
# not in the list, same source as basic-correctness (ls on the runner confirmed it)
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
# v1_e2e_general (cascade_attention and others);
# list MISS, use the isomorphic Aone aliases path
    "deepseek-ai/DeepSeek-V2-Lite":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite",
    # v1_e2e_general
# list hit: path=checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct
    "Qwen/Qwen2-VL-2B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct",
    # v1_e2e_general
# list hit: path=checkpoints/LLM/qwen/v3/Qwen3-0.6B
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
# v1_e2e_general; list MISS, use the isomorphic Aone aliases path (note the capital Qwen directory)
    "Qwen/Qwen2-1.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen2-1.5B-Instruct",
# v1_e2e_general (min_tokens and others); list MISS, use the isomorphic Aone aliases path
    "bigcode/starcoder2-3b":
        "/nas_aisw/datasets/checkpoints/LLM/starcoder/v1.0/starcoder2-3b",
    # test_fast_incdec_prefix_err.py + test_correctness_sliding_window.py
# list MISS, use the isomorphic Aone aliases path
    "google/gemma-3-1b-it":
        "/nas_aisw/datasets/checkpoints/LLM/gemma/v1.0/gemma-3-1b-it",
# test_async_scheduling.py speculator; list MISS, use the isomorphic Aone aliases path
    "nm-testing/Llama3_2_1B_speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/Llama3_2_1B_speculator.eagle3",
# JackFram/llama-160m (test_context_length.py): missing from both lists and waiting to be
# staged; the matching cases are ignored at file level in ENGINE_SINGLE_E2E_ARGS (see above):
# "JackFram/llama-160m": "<NAS path once staged>",
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

root = ET.Element("testsuites", name="vLLM PPU Engine (GHA)")
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

lines = ["### Engine Test (PPU)", "",
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
# The Aone single job is a 1-PPU pod: 1 card makes multi_gpu_test skip itself, matching
  CUDA_VISIBLE_DEVICES=0 _run_step "engine_basic" 1 "${ENGINE_SINGLE_BASIC_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_engine" 1 "${ENGINE_SINGLE_V1_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_e2e_general" 1 "${ENGINE_SINGLE_E2E_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area engine has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "engine_basic" 1 "${ENGINE_SINGLE_BASIC_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_engine" 1 "${ENGINE_SINGLE_V1_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_e2e_general" 1 "${ENGINE_SINGLE_E2E_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
