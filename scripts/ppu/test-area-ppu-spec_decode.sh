#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-spec_decode.sh -- PPU Spec Decode tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (inside the K8s worker pod,
#         cwd = /workspace/source).
#
# Environment variables:
#   TEST_MODE   all(default) | single   - the spec_decode area is single only (4 steps, each
#               with its own -k filter); no multi section (all 4 upstream buildkite steps are single)
#
# WARNING FIRST-DEPLOY RED EXPECTED (the first run is expected to fail everything):
#   most models the first version of this area depends on (Llama-4-Scout gated/heavy, the full
#   DeepSeek sizes, several eagle/mtp draft families) are not yet confirmed staged in /nas_aisw;
#   offline they fail with LocalEntryNotFoundError. This RED baseline is accepted and serves as
#   the regression verifier once staging completes (GREEN after staging = the PPU code is really
#   OK; a RED->GREEN flip catches a real regression).
#   Restore: it turns GREEN once the models marked pending staging in MODEL_MAP reach NAS.
#   Missing fixtures remain a failure in the daily full regression.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|all) ;;
  multi) echo "[mode] spec_decode area has no multi-mode steps (single-only)" >&2 ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch pytest_args passthrough: split on whitespace and append to
# the tail of every pytest command (e.g. `-k test_foo -x`) so a rerun can be narrowed without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-spec-decode-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area-specific dependencies: none.
# ------------------------------------------------------------------------------
# The spec_decode cases do not need ray (single-only, no distributed_executor_backend="ray").
# pytest-asyncio / tblib / pytest-shard are covered by the shared ppu_install_dependency.sh.
# suffix decoding needs arctic-inference (source dist only, the CI pod lacks the build toolchain
# to compile it), so the -k filter already excludes suffix_decoding_acceptance /
# speculative_config1 and nothing is installed here; restore = pre-install arctic-inference in the image.

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/spec_decode.yaml)
# ------------------------------------------------------------------------------
# Mirrors the 4 steps of upstream .buildkite/test_areas/spec_decode.yaml; v0.27 split them into
# the eagle/speculators+mtp/ngram_suffix/draft_model subdirectories. PPU merges them into 4
# single steps, one card each (nproc=1); draft_model/test_lora.py is only excluded from the
# draft_model step (not covered while Path E lora is paused, extras F12 follow-up). tp>1 cases
# skip automatically on 1 card.

# Step 1: eagle correctness
SD_EAGLE_ARGS=(
  tests/v1/e2e/spec_decode/eagle
)

# Step 2: speculators or mtp_correctness
#   excludes qwen3_eagle3_speculator / llama3_eagle3_speculator (2026-05-28 / 2026-06-01
#   smoke MR follow-up): test_speculators_model_integration asserts draft_model ==
#   'RedHatAI/...speculator.eagle3' and the like, but the vLLM alias redirect rewrites it to the
#   local /nas_aisw path -> the strings differ -> AssertionError.
#   Restore: once the upstream PR compares the basename instead of the full path.
SD_SPECULATORS_MTP_ARGS=(
  tests/v1/e2e/spec_decode/speculators
  tests/v1/e2e/spec_decode/mtp
  -k
  "not qwen3_eagle3_speculator and not llama3_eagle3_speculator"
)

# Step 3: ngram or suffix
#   excludes suffix_decoding_acceptance / speculative_config1: suffix decoding needs
#   arctic-inference (source dist only, needs nanobind+cmake to compile) and the CI PPU pod
#   lacks the build toolchain, so compilation fails (a user dev pod works). Restore: pre-install
#   arctic-inference in the image.
SD_NGRAM_SUFFIX_ARGS=(
  tests/v1/e2e/spec_decode/ngram_suffix
  -k
  "not suffix_decoding_acceptance and not speculative_config1"
)

# Step 4: draft_model + no_sync + batch_inference
#   excludes _tensor_parallelism / _engine_args / _no_sync_with_spec_decode (TP-arg
#   validation cases plus the Llama/DeepSeek model dependency of no_sync; Phase F+ F13 tracks
#   the PPU device detection path issue).
SD_DRAFT_MODEL_ARGS=(
  tests/v1/e2e/spec_decode/draft_model
  --ignore=tests/v1/e2e/spec_decode/draft_model/test_lora.py
  -k
  "not tensor_parallelism and not engine_args and not no_sync"
)

if [[ "${PPU_RELEASE:-}" == "v0.26.0" ]]; then
  SD_EAGLE_ARGS=(
    tests/v1/e2e/spec_decode/test_spec_decode.py
    -k
    "eagle"
  )
  SD_SPECULATORS_MTP_ARGS=(
    tests/v1/e2e/spec_decode/test_spec_decode.py
    -k
    "mtp or speculator"
  )
  SD_NGRAM_SUFFIX_ARGS=(
    tests/v1/e2e/spec_decode/test_spec_decode.py
    -k
    "(ngram or suffix) and not suffix_decoding_acceptance and not speculative_config1"
  )
  SD_DRAFT_MODEL_ARGS=(
    tests/v1/e2e/spec_decode/test_spec_decode.py
    tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py
    -k
    "not tensor_parallelism and not engine_args and not no_sync"
  )
elif [[ "${PPU_RELEASE:-}" != "v0.27.0" && "${PPU_RELEASE:-}" != "v0.28.0" ]]; then
  echo "[tests] ERROR: PPU_RELEASE must identify a supported release" >&2
  exit 2
fi

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
# === aliased / red-zone list hits (the symlink works; whether the path really exists on NAS is reported as MISS) ===
# Path source labels: (ckpt)=red-zone checkpoints_cleaned snapshot (main list)
#             (aone)=aone_ci/scripts/ppu_model_aliases.json (/ppusw/->/nas_aisw/)
# meta-llama/Llama-3.1-8B-Instruct: the extras explicitly reuse the Meta-Llama-3.1-8B-Instruct
# mirror (aliased on 2026-05-28, smoke MR #87 follow-up). (ckpt/aone agree)
    "meta-llama/Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Meta/v1.0/Meta-Llama-3.1-8B-Instruct",
# used by eagle_correctness / draft_model; the base plus the eagle draft family
    "meta-llama/Llama-3.2-1B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3.1-Instruct-8B",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE3/v3/EAGLE3-LLaMA3.1-Instruct-8B",
    "nm-testing/Llama3_2_1B_speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/"
        "Llama3_2_1B_speculator.eagle3",
# used by speculators / mtp / ngram; Qwen3 plus the eagle3 speculator family
    "Qwen/Qwen3-0.6B":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
    "Qwen/Qwen3-0.6B-FP8":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B-FP8",
    "Qwen/Qwen3-1.7B":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B",
    "Qwen/Qwen3-1.7B-FP8":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-8B":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-8B",
    "AngelSlim/Qwen3-1.7B_eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-1.7B_eagle3",
    "AngelSlim/Qwen3-8B_eagle3":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-8B_eagle3",
# Note: the qwen3/llama3 eagle3 speculator assertion cases are already deselected in -k
# (alias-rewrite AssertionError); the symlink stays for other cases that reference it.
    "RedHatAI/Qwen3-8B-speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3-8B-speculator.eagle3",
    "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/"
        "Llama-3.1-8B-Instruct-speculator.eagle3",
    "amd/PARD-Qwen3-0.6B":  # (aone) ngram/pard
        "/nas_aisw/datasets/checkpoints/LLM/PARD/v1.0/PARD-Qwen3-0.6B",
    "XiaomiMiMo/MiMo-7B-Base":  # (aone) mtp
        "/nas_aisw/datasets/checkpoints/LLM/MiMo/v1.0/MiMo-7B-Base",
# deepseek MTP / no_sync (small random variants, not the full DeepSeek)
    "ZixiQi/DeepSeek-V3-4layers-MTP-FP8":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V3/DeepSeek-V3-4layers-MTP-FP8",
    "eagle618/deepseek-v3-random":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/v3/deepseek-v3-random",
    "eagle618/eagle-deepseek-v3-random":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/eagle/v3/eagle-deepseek-v3-random",
# VL eagle3 (multimodal draft)
    "Qwen/Qwen2.5-VL-7B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct":  # (ckpt)
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-8B-Instruct",
    "Rayzl/qwen2.5-vl-7b-eagle3-sgl":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/qwen2.5/v1.0/qwen2.5-vl-7b-eagle3-sgl",
    "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct":  # (aone)
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/"
        "EAGLE-Llama-4-Scout-17B-16E-Instruct",

# === pending staging (MISS in both red-zone sources, no path guessing; these cases failing on the first run is the expected RED) ===
# not staged per the extras (gated / huge):
#   meta-llama/Llama-4-Scout-17B-16E-Instruct  gated + ~30 GB
#     (aone has a /llama/v4/ alias but staging is unconfirmed, the _heavy cases are already dropped by -k)
#   deepseek-ai/DeepSeek-R1                     huge, MISS in both sources
#   deepseek-ai/DeepSeek-V3                     huge (full no_sync, already dropped by -k)
# referenced by tests but MISS in both sources:
#   taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3
#   premjatin/qwen-linear-algebra-coder
#   likaixin/InstructCoder (a dataset, not a model; conftest redirect, see the extras)
# Restore: add the paths here once the models are staged in /nas_aisw (Phase F+ F10/F11).
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

root = ET.Element("testsuites", name="vLLM PPU Spec Decode (GHA)")
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

lines = ["### Spec Decode Test (PPU)", "",
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

# The spec_decode area is single only (4 steps, one single-card process each; mirrors the 4
# _run_pytest calls of the single section in aone test_area_ppu_spec_decode.sh).
# Limiting to 1 card (CUDA_VISIBLE_DEVICES=0) makes tp>1 cases skip automatically, matching a 1-PPU pod.
if [ "${MODE}" = "multi" ]; then
  echo "[run] spec_decode area has no multi-mode steps configured — nothing to run" >&2
else  # single | all
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_eagle" 1 "${SD_EAGLE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_speculators_mtp" 1 "${SD_SPECULATORS_MTP_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_ngram_suffix" 1 "${SD_NGRAM_SUFFIX_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "spec_decode_draft_model" 1 "${SD_DRAFT_MODEL_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
