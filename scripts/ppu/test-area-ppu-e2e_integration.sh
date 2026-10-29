#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-e2e_integration.sh -- PPU E2E Integration tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (inside the K8s worker pod,
#         cwd = /workspace/source).
#
# The cases use the GSM8K accuracy framework of tests/evals/gsm8k/test_gsm8k_correctness.py:
#   the config yaml defines model + server_args + accuracy_threshold and conftest.py
#   parametrizes over --config-list-file (configs live in tests/evals/gsm8k/configs/).
#
# Environment variables:
#   TEST_MODE          all(default) | single | multi
#   PPU_DEVICE_LABEL   device label (default OAM-810E); reproduces device_conditional_ignores
#
# Note (offline GSM8K data): tests/evals/gsm8k/gsm8k_eval.py downloads train/test.jsonl
#   from raw.githubusercontent.com into /tmp at run time. The red-zone pod has no internet,
#   so the [setup] pre-cache block below seeds /tmp/{train,test}.jsonl from the /nas_aisw
#   candidates (same as ppu_ci_area_lm_eval.sh). Without those files it warns and the
#   GSM8K cases fail with ConnectionError (restore: stage the jsonl into a candidate path).
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

# Device label (the workflow injects PPU_DEVICE_LABEL=OAM-810E through extra_env); mirrors
# the aone sh's second argument and drives device_conditional_ignores (OAM-810E skips FP8).
DEVICE_LABEL="${PPU_DEVICE_LABEL:-OAM-810E}"
echo "[device] label: ${DEVICE_LABEL}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-e2e-integration-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area-specific dependencies: none.
# ------------------------------------------------------------------------------
# The GSM8K framework dependencies (aiohttp/numpy/regex/requests/tqdm/pyyaml) ship in the
# image; the pytest toolchain (pytest-asyncio/tblib/pytest-shard/pyyaml) is installed by
# ppu_ci_nightly_install_dependency.sh (mirroring the legacy pip install pytest-asyncio tblib
# pytest-shard), so nothing is added here.

# ------------------------------------------------------------------------------
# [setup] pre-cache the GSM8K jsonl (the red-zone pod cannot reach raw.githubusercontent.com)
# ------------------------------------------------------------------------------
# gsm8k_eval.py:download_and_cache_file() pulls {train,test}.jsonl from GitHub raw into
# /tmp/{train,test}.jsonl with requests.get (an existing file is reused).
# With no internet the red zone raises ConnectionError and every GSM8K case fails.
# HF_HUB_OFFLINE only hooks huggingface_hub and does not affect requests, so the download
# cannot be avoided. Same logic as the lm-eval script: probe and copy from the /nas_aisw
# candidates and warn when missing (the cases then fail with the reason in the log).
# Added in run 34594819557: this script used to miss the block (lm-eval had it) and both
# e2e steps always failed.
echo "========== [setup] pre-cache gsm8k jsonl =========="
for f in train.jsonl test.jsonl; do
  dst="/tmp/${f}"
  if [ -f "${dst}" ]; then
    echo "[gsm8k] ${dst} already present"
    continue
  fi
  for _cand in \
    "/nas_aisw/datasets/gsm8k/${f}" \
    "/nas_aisw/datasets/eval/gsm8k/${f}" \
    "/nas_aisw/datasets/grade-school-math/${f}"; do
    if [ -f "${_cand}" ]; then
      cp "${_cand}" "${dst}" && echo "[gsm8k] cached ${f} <- ${_cand}"
      break
    fi
  done
  if [ ! -f "${dst}" ]; then
echo "[gsm8k] WARN: ${f} not found on /nas_aisw candidates - GSM8K tests will fail (not staged yet)"
  fi
done

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single/multi sections in aone_ci/ppu_extras/e2e_integration.yaml)
# ------------------------------------------------------------------------------
# Fixed in run 34594819557: models-ppu-e2e-{single,multi,fp8}.txt and the 3 yamls they
# reference (DeepSeek-V2-Lite-Prefetch-Offload / DeepSeek-V2-Lite-EP-EPLB /
# Qwen3-30B-A3B-FP8-EP-EPLB) are now in tests/evals/gsm8k/configs/ (copied from the
# verified Aone version). Before that 7 files were missing, so conftest had nothing to
# parametrize and collected a bare test_gsm8k_correctness that reported
# fixture 'config_filename' not found (both steps failed).
# single = the Aone "e2e-integration single" job (1-PPU pod):
#   - e2e_prefetch_offload: DeepSeek-V2-Lite prefetch offload accuracy (1 GPU);
#     GSM8K eval framework, threshold=0.25, 200 questions (the single section pins
#     CUDA_VISIBLE_DEVICES=0, matching the Aone 1-PPU pod)
E2E_SINGLE_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-single.txt
)

# multi = the Aone "e2e-integration multi" job (4-PPU pod), two steps in order:
#   - e2e_ep_eplb: DeepSeek-V2-Lite EP+EPLB accuracy (4 GPU, TP=2 DP=2);
#     GSM8K eval framework, threshold=0.25, 200 questions
E2E_MULTI_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-multi.txt
)
#   - e2e_fp8_ep_eplb: Qwen3-30B-A3B-FP8 EP+EPLB accuracy (4 GPU, FP8);
#     OAM-810E does not support FP8, so per the original yaml device_conditional_ignores
#     the whole file is --ignore'd when DEVICE_LABEL=OAM-810E (handled by the dispatch below).
E2E_FP8_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-fp8.txt
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
# e2e_prefetch_offload (single) + e2e_ep_eplb (multi): DeepSeek-V2-Lite.
# Not in the list; the Aone ppu_model_aliases.json
# hits /ppusw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite
# (/ppusw -> /nas_aisw).
    "deepseek-ai/DeepSeek-V2-Lite":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite",
# e2e_fp8_ep_eplb (multi): Qwen3-30B-A3B-FP8. OAM-810E has no FP8, so this step is
# whole-file ignored on this device and the model never loads; the path stays registered
# for later. List hit: checkpoints_cleaned.json path=checkpoints/LLM/qwen/v3/Qwen3-30B-A3B-FP8
    "Qwen/Qwen3-30B-A3B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-30B-A3B-FP8",
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

root = ET.Element("testsuites", name="vLLM PPU E2E Integration (GHA)")
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

lines = ["### E2E Integration Test (PPU)", "",
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

# ------------------------------------------------------------------------------
# [run/skip] e2e_fp8_ep_eplb: reproduces the aone sh device_conditional_ignores.
# OAM-810E has no FP8, so the aone sh skips the whole file with --ignore (0 cases
# collected). Here it is recorded as skipped (a skipped junit entry + label registration),
# which keeps the skip intent, avoids pytest "no tests collected" (exit 5) failing the
# whole round, and still lets other devices run the step through _run_step.
# ------------------------------------------------------------------------------
_run_fp8_step() {
  local label="e2e_fp8_ep_eplb"
  case "${DEVICE_LABEL}" in
    OAM-810E)
echo "========== [step] ${label} (skipped: OAM-810E does not support FP8) =========="
      STEP_LABELS_LIST="${STEP_LABELS_LIST} ${label}"
      python3 - "${TMP_JUNIT}/${label}.xml" "${label}" <<'PYEOF'
import sys
from xml.etree import ElementTree as ET

out, label = sys.argv[1], sys.argv[2]
ts = ET.Element("testsuite", name=label, tests="1", errors="0",
                failures="0", skipped="1", time="0")
tc = ET.SubElement(ts, "testcase", name=label,
                   classname=f"gha_ci.{label}", time="0")
ET.SubElement(tc, "skipped",
              message="OAM-810E does not support FP8 "
                      "(device_conditional_ignores)")
ET.ElementTree(ts).write(out, encoding="UTF-8", xml_declaration=True)
print(f"[step] {label} skipped (device={out})")
PYEOF
      ;;
    *)
      _run_step "${label}" 1 "${E2E_FP8_ARGS[@]}"
      ;;
  esac
}

if [ "${MODE}" = "single" ]; then
# The Aone single job is a 1-PPU pod: DeepSeek-V2-Lite prefetch offload runs on 1 card, so
# CUDA_VISIBLE_DEVICES=0 matches the Aone 1-PPU pod
  CUDA_VISIBLE_DEVICES=0 _run_step "e2e_prefetch_offload" 1 "${E2E_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
# e2e_ep_eplb: DeepSeek-V2-Lite TP=2 DP=2 uses all 4 cards (no CUDA_VISIBLE_DEVICES limit)
  _run_step "e2e_ep_eplb" 1 "${E2E_MULTI_ARGS[@]}"
  _run_fp8_step
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "e2e_prefetch_offload" 1 "${E2E_SINGLE_ARGS[@]}"
  _run_step "e2e_ep_eplb" 1 "${E2E_MULTI_ARGS[@]}"
  _run_fp8_step
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
