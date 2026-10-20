#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-benchmarks.sh - PPU Benchmarks test runner (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/test-area-ppu-benchmarks.yml (in the container,
# cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all (default) | single - this area is single-only
#
# Upstream counterpart: .buildkite/test_areas/benchmarks.yaml (3 steps). The PPU
# scope only takes "Benchmarks CLI Test" -> pytest tests/benchmarks/; the other
# two steps cannot run in the red zone:
#   - Benchmarks (run-benchmarks.sh): downloads Llama-2-7b + the ShareGPT
#     dataset at runtime, and the red zone has no internet access -> excluded
#   - Attention Benchmarks Smoke Test: B200-only -> excluded
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|all) ;;
  multi) echo "[mode] ERROR: area benchmarks has no multi-mode steps configured" >&2; exit 2 ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# pytest_args passthrough from workflow_call: split on whitespace and appended
# to every step's pytest command (e.g. `-k test_foo -x`) to narrow the scope
# while debugging without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-benchmarks-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] Area-specific dependency: pandas
# ------------------------------------------------------------------------------
if python3 -c "import pandas" 2>/dev/null; then
  echo "[deps] pandas already installed: $(python3 -c 'import pandas; print(pandas.__version__)')"
else
  echo "[deps] installing pandas from flytiger PyPI"
  python3 -m pip install --no-cache-dir pandas \
    -i "https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
fi

# ------------------------------------------------------------------------------
# [tests] Test selection (snapshot of the single stage in
# legacy_ci/ppu_extras/benchmarks.yaml)
# ------------------------------------------------------------------------------
# single = legacy CI "benchmarks single" job (1-PPU pod):
#   tests/benchmarks/ - benchmark CLI unit tests + argument parsing tests.

# The two files below are ignored: they need the gpt2 tokenizer, which is not
# available offline in the red zone; run 49488451 confirmed 18 ERROR results
# from these two files in a full run:
#   - test_random_dataset.py                  needs gpt2 tokenizer offline
#   - test_random_multimodal_dataset_video.py needs cv2 + gpt2 tokenizer
# Restore once the gpt2 tokenizer is added to the red zone: drop the matching
# --ignore (and extend MODEL_MAP).
BENCH_SINGLE_ARGS=(
  tests/benchmarks/
  "--ignore=tests/benchmarks/test_random_dataset.py"
  "--ignore=tests/benchmarks/test_random_multimodal_dataset_video.py"
  # Run 38721031031: the Namespace hand-built by the test lacks arguments added
  # in v0.28. Remove this entry once the upstream test adds
  # skip_first_concurrency.
  "--deselect=tests/benchmarks/test_skip_tokenizer_init.py::test_main_async_skip_tokenizer_init_does_not_raise"
)

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
# the list follow the legacy CI /ppusw layout and are marked "to be confirmed".
# Missing paths are reported as WARN and skipped.
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # test_latency_cli.py / test_throughput_cli.py / test_serve_cli.py:
    # `vllm bench {latency,throughput,serve} --model meta-llama/Llama-3.2-1B-Instruct
    # --load-format dummy` - dummy weights are not downloaded, but the model
    # config/tokenizer are still needed, so a local checkpoint is required.
    # List hit: checkpoints_cleaned.json path=checkpoints/LLM/Llama/v3.2/
    # Llama-3.2-1B-Instruct (the list's ms_name is unsloth/..., matched by name;
    # legacy CI /ppusw uses a v3.1 directory, but the red-zone list's v3.2 wins,
    # consistent with basic-correctness)
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

root = ET.Element("testsuites", name="vLLM PPU Benchmarks (GHA)")
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

lines = ["### Benchmarks Test (PPU)", "",
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

# benchmarks is single-only (legacy CI extras only have a single stage): both
# single and all run the same benchmarks_cli step; multi is rejected by the case
# statement above.
_run_step "benchmarks_cli" 1 "${BENCH_SINGLE_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] Aggregate exit code (script exit code = aggregate rc, so the CI
# signal stays accurate)
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
