#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-weight_loading.sh -- PPU Weight Loading tests (GitHub Actions)
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
TMP_JUNIT="/tmp/ppu-weight-loading-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the multi section in aone_ci/ppu_extras/weight_loading.yaml)
# ------------------------------------------------------------------------------
# Upstream weight_loading iterates models.txt (27 model configs) with
# run_model_weight_loading_test.sh; each config drives the same
# test_weight_loading.py through env vars (MODEL_NAME/QUANTIZATION/REVISION)
# and that file hardcodes tensor_parallel_size=2. PPU replicates the mechanism
# with per-config env vars and a single-process pytest.
#
# Red-zone selection (2 configs, both TP=2, both aliased on NAS and green in run 34580922394):
#   wl_gptq_marlin_zephyr    gptq_marlin  robertgshaw2/zephyr-7b-beta-channelwise-gptq
#   wl_none_fairseq2         None         mgleize/fairseq2-dummy-Llama-3.2-1B
#
# 4 configs removed in run 34580922394:
#   - wl_gptq_marlin_tinyllama / wl_gptq_tinyllama
#     (TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ): not staged on NAS, [setup] MISS ->
#     LocalEntryNotFoundError right away. To restore, add it back once staged.
#   - wl_ct_w8w8_static / wl_ct_w8a8_dynamic (nm-testing/tinyllama-oneshot-*,
#     compressed-tensors INT8): deep_gemm int8 shape assert
#     (lhs_scales.shape[0] == m / rhs_scales.shape[0] == n) and the PPU int8
#     scaled_mm kernel does not support it. To restore, add it back once fixed.
#
# Excluded (from the original yaml availability analysis; none meet the red-zone conditions):
#   - fp8 models (nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test,
#     neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV): aliased, but PPU OAM-810E is
#     SM8.0 < the SM8.9 that FP8 needs, skipped.
#   - not staged and unaliased (TheBloke/Llama-2-7B-GPTQ, TechxGenus/gemma-1.1-2b-it-GPTQ,
#     nm-testing/tinyllama-oneshot-w4a16-group128-v2,
#     nm-testing/tinyllama-oneshot-w8a16-per-channel,
#     nm-testing/Phi-3-mini-128k-instruct-FP8,
#     neuralmagic/Phi-3-medium-128k-instruct-quantized.w4a16,
#     casperhansen/mixtral-instruct-awq, ...): no staged checkpoint in the red zone, skipped.
#   - the SparseLlama 2of4 group with capability>=90: unaliased and needs SM9.0, skipped.
#   - revision variants (gptq-8bit-32g, ...): model redirect maps one path to the
#     main weights only; variants need their own staged checkpoint, so only main runs.
#
# Module-level snapshot_download audit: CLEAN - test_weight_loading.py reads the
# model name from env vars and downloads nothing at import time; vllm_runner loads it.
WL_LABELS=(
  wl_gptq_marlin_zephyr
  wl_none_fairseq2
)
WL_QUANT=(
  gptq_marlin
  None
)
WL_MODEL=(
  robertgshaw2/zephyr-7b-beta-channelwise-gptq
  mgleize/fairseq2-dummy-Llama-3.2-1B
)
WL_REV=(main main)
WL_ARGS=(tests/weight_loading/test_weight_loading.py)

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
    # Paths come from aone_ci/scripts/ppu_model_aliases.json (/ppusw/ -> /nas_aisw/).
    # These weight_loading-only quantized models are missing from the main list, so
    # the NAS path is assembled from the Aone alias table.
    # wl_gptq_marlin_zephyr
    "robertgshaw2/zephyr-7b-beta-channelwise-gptq":
        "/nas_aisw/datasets/checkpoints/LLM/zephyr/v1.0/"
        "zephyr-7b-beta-channelwise-gptq",
    # wl_none_fairseq2
    "mgleize/fairseq2-dummy-Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/fairseq2/v1.0/"
        "fairseq2-dummy-Llama-3.2-1B",
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

root = ET.Element("testsuites", name="vLLM PPU Weight Loading (GHA)")
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

lines = ["### Weight Loading Test (PPU)", "",
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

# Every weight_loading case is TP=2 (multi section): single has no step, while
# multi and all run both model configs. Each config passes MODEL_NAME/
# QUANTIZATION/REVISION as a command-prefix env var (inherited by the pytest
# child) in a single process (shards=1, no CUDA_VISIBLE_DEVICES limit -> both
# cards visible, TP=2 uses them).
_run_weight_loading_configs() {
  local i
  for i in "${!WL_LABELS[@]}"; do
    MODEL_NAME="${WL_MODEL[$i]}" QUANTIZATION="${WL_QUANT[$i]}" \
      REVISION="${WL_REV[$i]}" \
      _run_step "${WL_LABELS[$i]}" 1 "${WL_ARGS[@]}"
  done
}

if [ "${MODE}" = "single" ]; then
  echo "[run] no single steps configured for this area (all cases are TP=2)"
elif [ "${MODE}" = "multi" ]; then
  _run_weight_loading_configs
else  # all
  _run_weight_loading_configs
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
