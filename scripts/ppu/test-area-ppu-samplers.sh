#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-samplers.sh - PPU Samplers test runner (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/test-area-ppu-samplers.yml (in the container,
# cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all (default) | single - this area has no multi stage
#               (tests/samplers/ has no @multi_gpu_test; single PPU only)
#
# Known follow-up:
#   - Flashinfer variant: upstream runs twice (the 2nd time with
#     VLLM_USE_FLASHINFER_SAMPLER=1). The legacy CI sh_renderer has no per-step
#     env field, so it was not split; on GHA a 2nd step can be added if needed
#     (just export the env at the _run_step call site).
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# pytest_args passthrough from workflow_call: split on whitespace and appended
# to every step's pytest command (e.g. `-k test_foo -x`) to narrow the scope
# while debugging without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-samplers-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] Test selection (snapshot of the single stage in
# legacy_ci/ppu_extras/samplers.yaml)
# ------------------------------------------------------------------------------
# samplers_default = PPU version of the upstream "Samplers Test" step (runs 2
# of its 4 files):
SAMPLERS_SINGLE_ARGS=(
  tests/samplers
  # test_beam_search.py: TinyLlama/TinyLlama-1.1B-Chat-v1.0 is not staged
  # (original yaml comment). Note: the model is now in the red-zone list
  # (checkpoints/LLM/TinyLlama/v1.0/...) and MODEL_MAP already links it; beam
  # search correctness on PPU is not verified yet, so drop this line once the
  # first runs are stable
  --ignore=tests/samplers/test_beam_search.py
  # test_no_bad_words.py (2026-05-28, original yaml comment): setup_method
  # calls transformers AutoTokenizer.from_pretrained(MODEL) directly, bypassing
  # vLLM's maybe_model_redirect -> red-zone cache miss on legacy CI -> 2/8 fail.
  # GHA uses symlinks in the standard HF cache layout (see MODEL_MAP below), so
  # from_pretrained reads the cache offline without going through the vllm
  # redirect and the original gap does not exist structurally. The NAS path of
  # distilgpt2 is not confirmed by the list (same layout as the legacy CI
  # aliases), so this version still ignores it to be safe; drop this line to
  # verify once the first run confirms symlink MISS/OK
  --ignore=tests/samplers/test_no_bad_words.py
)

# multi: none (the samplers area has no multi_gpu_test; single PPU only)

# ------------------------------------------------------------------------------
# [env] Offline + runtime configuration
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# Offline by default (models come from the preloaded /nas_aisw volume); set
# PPU_TEST_ONLINE=1 when online downloads are needed
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF cache: the workflow injects HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub;
# when it is not injected (local debugging), probe candidate paths under /nas_aisw
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor compilation needs cuda.h + ptxas + libcuda (without
# them torch.compile tests fail with BackendCompilerFailed)
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [setup] HF hub cache symlinks: HF id -> local checkpoint on /nas_aisw
# ------------------------------------------------------------------------------
# Paths prefer the path field of the full-branch model list of models stored in
# the red zone (absolute NAS path = /nas_aisw/datasets/ + path); models not in
# the list use the same layout as legacy CI's ppu_model_aliases.json
# (/ppusw/ -> /nas_aisw/). Missing paths are reported as WARN and skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # test_logprobs.py + test_ignore_eos.py
    # Missing from the list; use the same layout as the legacy CI aliases
    "distilbert/distilgpt2":
        "/nas_aisw/datasets/checkpoints/LLM/gpt/v1.0/distilgpt2",
    # test_ignore_eos.py (legacy CI used to redirect it to Instruct; the GHA
    # symlink maps the real name directly, list hit
    # path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B)
    "meta-llama/Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B",
    # test_beam_search.py (currently ignored; the model is in the list and is
    # used once re-enabled): list hit
    # path=checkpoints/LLM/TinyLlama/v1.0/TinyLlama-1.1B-Chat-v1.0
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/TinyLlama-1.1B-Chat-v1.0",
    # test_no_bad_words.py (currently ignored; used once re-enabled); same
    # source as basic-correctness (the runner confirmed it exists via ls)
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
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
# [junit] Merge infrastructure (including a fallback for pytest crashes)
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # only called via `trap _emit_junit EXIT` below
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

root = ET.Element("testsuites", name="vLLM PPU Samplers (GHA)")
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

# ---- step summary: per-shard stats table (markdown)
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

lines = ["### Samplers Test (PPU)", "",
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
  # pytest output of sharded steps goes to TMP_JUNIT; collect it into the
  # artifact as well for debugging
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  # The container runs as root; artifacts must be readable by the runner user
  # (upload-artifact)
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] Return the per-unit stats table via NAS: RESULTS_DIR lives in
  # the Pod and is lost once the Pod is reclaimed.
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ] && [ -f "${RESULTS_DIR}/summary.md" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -f "${RESULTS_DIR}/summary.md" "${PPU_SUMMARY_NAS_DIR}/summary.md"; then
      echo "[k8s-summary] summary.md -> ${PPU_SUMMARY_NAS_DIR}/summary.md"
    else
      echo "[k8s-summary] WARN: failed to return summary via NAS (${PPU_SUMMARY_NAS_DIR} not writable?)"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] Single-step runner: shards>1 -> concurrent shards; shards==1 -> single process
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
  CUDA_VISIBLE_DEVICES=0 _run_step "samplers_default" 1 "${SAMPLERS_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area samplers has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "samplers_default" 1 "${SAMPLERS_SINGLE_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] Aggregate exit code (script exit code = aggregate rc, so the CI
# signal stays accurate)
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
