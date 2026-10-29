#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-lora.sh -- PPU LoRA tests (GitHub Actions)
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
TMP_JUNIT="/tmp/ppu-lora-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single/multi sections in aone_ci/ppu_extras/lora.yaml)
# ------------------------------------------------------------------------------
# single = the PPU version of the upstream "LoRA %N" job (parallelism=4):
#   - drops the 7 TP files (they run in the multi section) plus 3 files PPU cannot run
#   - OAM-810E device ignores (snapshot of device_conditional_ignores):
#     the fp8 kernel needs SM>=8.9; triton fused_moe accuracy problems. When a new device
#     is onboarded, add or remove the two lines below per the per-device semantics of lora.yaml.
LORA_SINGLE_ARGS=(
  tests/lora
  --ignore=tests/lora/test_chatglm3_tp.py
  --ignore=tests/lora/test_llama_tp.py
  --ignore=tests/lora/test_qwen3_with_multi_loras.py
  --ignore=tests/lora/test_olmoe_tp.py
  --ignore=tests/lora/test_deepseekv2_tp.py
  --ignore=tests/lora/test_gptoss_tp.py
  --ignore=tests/lora/test_qwen3moe_tp.py
# Upstream v0.23.0 also excludes this file from the `LoRA %N` step (it belongs to the
# 4-card LoRA TP step); the old snapshot missed it, restored here to match upstream.
  --ignore=tests/lora/test_qwen35_densemodel_lora.py
# Note: upstream v0.23.0 does NOT ignore this file (it runs in the sharded step); the
# PPU side keeps excluding it as the old snapshot did - an unconfirmed extra exclusion.
# Ledger: scripts/ppu/docs/ppu-ci-exclusions.md, decide restore or add a reason once confirmed.
  --ignore=tests/lora/test_minicpmv_tp.py
# ChatGLM3 tokenizer._pad() lacks the padding_side argument and is incompatible with transformers
  --ignore=tests/lora/test_add_lora.py
# whisper-small base model - whisper support on PPU is unverified
  --ignore=tests/lora/test_whisper.py
# module-level snapshot_download; collection crashes offline (landmine)
  --ignore=tests/lora/test_default_mm_loras.py
# PPU inference accuracy differs, the assertion pattern does not match, PPU team to investigate
  -k "not test_qwen2vl_multiple_lora_types"
  # --- OAM-810E device ignores ---
# SM 8.0; the fp8e4m3fn kernel needs SM >= 8.9
  --ignore=tests/lora/test_punica_ops_fp8.py
# triton fused_moe kernel accuracy: Tensor mismatch 100%
  --ignore=tests/lora/test_fused_moe_lora_kernel.py
# --- known failures (platform/model issues, confirmed in the first full round on 08-20 and
# reproduced on 08-24 after reverting the self-hosted docker mode, still tracked) ---
# AWQ quantized inference fails the output assertion; model1 of the GPTQ variant in the same
# function passes, so model0 is excluded by nodeid (--deselect leaves other parametrizations alone)
  "--deselect=tests/lora/test_quant_model.py::test_quant_model_lora[model0]"
# Qwen2-VL engine init fails (Engine core init); the qwen25vl/qwen3vl cases in the same
# file pass. -k cannot be used: "test_qwen2vl_lora" is a substring prefix of
# test_qwen2vl_lora_beam_search and would take it down as well
  "--deselect=tests/lora/test_qwenvl.py::test_qwen2vl_lora"
# fixture qwen3moe_lora_files misses the offline cache (ERROR at setup):
# snapshot_download(jeeejeee/qwen3-moe-text2sql-spider) fails.
# MODEL_MAP has an entry but the directory is actually missing on NAS (list hit != NAS hit).
# The file holds a single test function (test_moe_lora_ep2_real_qwen3moe[0]/[1], a pure CPU
# EP slice assertion with no platform risk), so the whole file is ignored; delete this line
# once the adapter is staged on NAS.
  --ignore=tests/lora/test_moe_lora_ep_load.py
)

# multi = the PPU version of the upstream "LoRA TP (Distributed)" job (currently a single
# file, TP=2; the remaining TP files wait until their models are staged on /nas_aisw)
# -k excludes two TP cases for now: engine init fails on PPU - TP worker rank1 reports a
# CUDA OOM while allocating 20 MiB with 95.51 GiB free (false OOM). expandable_segments
# has been disabled (see [env] below, the prime suspect: the bc single uni section shows
# the same symptom, Aone does not set it and is all green); drop the -k below once a rerun
# confirms the false OOM is gone
LORA_MULTI_ARGS=(
  tests/lora/test_qwen3_with_multi_loras.py
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
# The model names in tests/lora are hard-coded HF ids; offline from_pretrained uses the
# standard HF_HUB_CACHE layout (models--org--repo/snapshots/main). The checkpoints staged
# on NAS live under /nas_aisw/datasets/checkpoints/LLM/ grouped by family, so symlinks in
# the HF layout are created for the models the tests need (no fork-specific mechanism).
# A missing path warns and is skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # ---- base models ----
# single/multi main models (conftest, test_lora_functions, test_qwen3_unembed, test_worker,
# test_qwen3_with_multi_loras)
# list hit: checkpoints_cleaned.json path=checkpoints/LLM/qwen/v3/Qwen3-0.6B
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
# MoE base of test_mixtral.py (list hit LLM/Mistral/v1)
    "mistralai/Mixtral-8x7B-Instruct-v0.1":
        "/nas_aisw/datasets/checkpoints/LLM/Mistral/v1/Mixtral-8x7B-Instruct-v0.1",
# the three VL bases of test_qwenvl.py (list hit LLM/qwen/{v2.0,v2.5,v3.0})
    "Qwen/Qwen2-VL-2B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-4B-Instruct",
# base of test_qwen35_densemodel_lora.py (list hit id 669
# checkpoints/LLM/Qwen/v1.0/Qwen3.5-4B - note the capitalised Qwen directory, unlike the
# lowercase qwen of the VL family above; NAS mixes both cases, so copy the list path as-is)
    "Qwen/Qwen3.5-4B":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3.5-4B",
    # ---- lora adapters ----
# paths taken from the red-zone list's lora_adapters snapshot (the runner measured them by
# finding adapter_config.json); when missing, [setup] warns and skips, the related cases
# fail and the log shows why
    "Jackmin108/Qwen3-0.6B-Meow-LoRA":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen3/v1.0/Qwen3-0.6B-Meow-LoRA",
    "Jackmin108/Qwen3-0.6B-Woof-LoRA":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen3/v1.0/Qwen3-0.6B-Woof-LoRA",
# EP slice cases of test_moe_lora_ep_load.py (list hit Qwen3/v1.0)
    "jeeejeee/qwen3-moe-text2sql-spider":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen3/v1.0/qwen3-moe-text2sql-spider",
# HF org=charent, NAS actually keeps it under LLM/self/ (tests/lora reference it 3+2 times)
    "charent/self_cognition_Alice":
        "/nas_aisw/datasets/checkpoints/LLM/self/v1.0/self_cognition_Alice",
    "charent/self_cognition_Bob":
        "/nas_aisw/datasets/checkpoints/LLM/self/v1.0/self_cognition_Bob",
    # test_mixtral.py / test_quant_model.py / test_transformers_model.py
    "SangBinCho/mixtral-lora":
        "/nas_aisw/datasets/checkpoints/LLM/mixtral/v1.0/mixtral-lora",
    "jashing/tinyllama-colorist-lora":
        "/nas_aisw/datasets/checkpoints/LLM/tinyllama/v1.0/tinyllama-colorist-lora",
    "jeeejeee/ilama-text2sql-spider":
        "/nas_aisw/datasets/checkpoints/LLM/ilama/v1.0/ilama-text2sql-spider",
    # test_lora_checkpoints.py / test_lora_huggingface.py /
# test_peft_helper.py: loads the checkpoint on CPU, only the adapter body is needed, not the base
    "jeeejeee/baichuan7b-text2sql-spider":
        "/nas_aisw/datasets/checkpoints/LLM/baichuan7b/v1.0/baichuan7b-text2sql-spider",
    "jeeejeee/baichuan7b-zero-init":
        "/nas_aisw/datasets/checkpoints/LLM/baichuan7b/v1.0/baichuan7b-zero-init",
    "jeeejeee/baichuan-7b-lora-zero-regex":
        "/nas_aisw/datasets/checkpoints/LLM/baichuan/v1.0/baichuan-7b-lora-zero-regex",
    "jeeejeee/chatglm3-text2sql-spider":
        "/nas_aisw/datasets/checkpoints/LLM/chatglm3/v1.0/chatglm3-text2sql-spider",
    "jeeejeee/llama32-3b-text2sql-spider":
        "/nas_aisw/datasets/checkpoints/LLM/llama32/v1.0/llama32-3b-text2sql-spider",
# VL adapters of test_qwenvl.py
    "jeeejeee/qwen2-vl-lora-pokemon":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/qwen2-vl-lora-pokemon",
    "jeeejeee/qwen25-vl-lora-pokemon":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/qwen25-vl-lora-pokemon",
# NAS has no separate tower directory; it shares the tower-connector one
# (the Aone aliases map the same way, not a typo)
    "prashanth058/qwen2vl-flickr-lora-tower":
        "/nas_aisw/datasets/checkpoints/LLM/qwen2vl/v1.0/qwen2vl-flickr-lora-tower-connector",
    "prashanth058/qwen2vl-flickr-lora-tower-connector":
        "/nas_aisw/datasets/checkpoints/LLM/qwen2vl/v1.0/qwen2vl-flickr-lora-tower-connector",
    "prashanth058/qwen2vl-flickr-lora-language":
        "/nas_aisw/datasets/checkpoints/LLM/qwen2vl/v1.0/qwen2vl-flickr-lora-language",
    "EpochEcho/qwen2.5-3b-vl-lora-vision-connector":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/qwen2.5-3b-vl-lora-vision-connector",
    "EpochEcho/qwen3-4b-vl-lora-vision-connector":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/qwen3-4b-vl-lora-vision-connector",
# ---- not in the list: guessed from the isomorphic Aone /ppusw -> /nas_aisw path, watch MISS on the first run ----
# quantized base of test_quant_model.py (the list only has the v1.0 non-quantized one)
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v0.3/TinyLlama-1.1B-Chat-v0.3-AWQ",
    "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v0.3/TinyLlama-1.1B-Chat-v0.3-GPTQ",
# base of test_transformers_model.py
    "hmellor/Ilama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Ilama/v3.2/Ilama-3.2-1B",
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

root = ET.Element("testsuites", name="vLLM PPU LoRA (GHA)")
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

lines = ["### LoRA Test (PPU)", "",
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
  _run_step "lora_full" 4 "${LORA_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  _run_step "test_qwen3_with_multi_loras" 1 "${LORA_MULTI_ARGS[@]}"
else  # all
  _run_step "lora_full" 4 "${LORA_SINGLE_ARGS[@]}"
  _run_step "test_qwen3_with_multi_loras" 1 "${LORA_MULTI_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
