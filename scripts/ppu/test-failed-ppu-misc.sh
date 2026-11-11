#!/bin/bash
# ==============================================================================
# scripts/ppu/test-failed-ppu-misc.sh -- rerun the v0.27.0 misc-area failures from
# nightly run 41978966338 (Nightly PPU test #18).
# Caller: .github/workflows/test_ppu_nightly.yml (in the container,
# cwd=/workspace/source, after ppu_ci_nightly_install_dependency.sh).
#
# Failures (20 on v0.27.0) and fixes under verification:
#   1. tests/v1/spec_decode/test_ngram.py fails at COLLECTION time:
#      ngram_proposer imports numba, requirements/ppu.txt keeps numba commented
#      out and the leftover image numba is built against the numpy 1.x ABI, so
#      the import dies once the env ships numpy 2.x. Same root cause as
#      v0.26.0 (fixed there in PR #57): ppu_ci_nightly_install_dependency.sh
#      uninstalls the image numba and installs numba==0.65.0 from the aliyun
#      mirror (the artifactory numba==0.65.0 is a PPU source wrapper with no
#      cp312 / ubuntu2404 / runtime 130 build).
#   2. tests/utils_/test_gpu_sync_debug.py (3 cases) died with "Cannot
#      re-initialize CUDA in forked subprocess": the tests used the default
#      fork start method of create_new_process_for_each_test while earlier
#      tests in the same pytest process already initialized CUDA. They now
#      request "spawn".
#   3. tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer (16 cases)
#      failed with "[Errno 22] Invalid argument" from
#      madvise(MADV_POPULATE_WRITE), which the PPU host kernel rejects.
#      Backport of upstream ef2615c2e: SharedOffloadRegion falls back to
#      touching the pages when MADV_POPULATE_WRITE is unsupported, with its
#      own unit tests in test_shared_offload_region.py (rerun here as well).
# A skipped case does not count as verified: the step rc becomes 1 and the
# summary marks the unit SKIPPED.
#
# Modeled on scripts/ppu/test-area-ppu-misc.sh of the ci/ppu-nightly-all-areas
# branch (env blocks, junit/summary emission); only the failing file runs here.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# workflow_dispatch pytest_args passthrough (kept for parity with the area script)
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

# ------------------------------------------------------------------------------
# [tests] exactly the cases that failed in run 41978966338 on v0.27.0
# ------------------------------------------------------------------------------
MISC_FAILED_NGRAM_ARGS=(
  tests/v1/spec_decode/test_ngram.py
)

MISC_FAILED_GPU_SYNC_DEBUG_ARGS=(
  "tests/utils_/test_gpu_sync_debug.py::test_with_env_set[error]"
  "tests/utils_/test_gpu_sync_debug.py::test_with_env_set[warn]"
  "tests/utils_/test_gpu_sync_debug.py::test_without_env_set"
)

MISC_FAILED_KV_OFFLOAD_ARGS=(
  # the MADV_POPULATE_WRITE fallback tests added by the ef2615c2e backport
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_madvise_success_selects_madvise_population"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_madvise_einval_selects_fallback_for_ranked_region"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_madvise_einval_selects_fallback_for_unranked_region"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_fallback_populate_write_preserves_bytes_and_faults_pages"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_madvise_unexpected_oserror_propagates"
)
# shared-memory cases only: [use_shared_memory=True-replicated_layout-device-seed-
# num_tensors-num_cpu_blocks-num_gpu_blocks-blocks_per_chunk-gpu_page_size_bytes-
# num_mappings-gpu_to_cpu]
for _replicated in False True; do
  for _blocks_per_chunk in 1 3; do
    for _page_size in 512 1024; do
      for _gpu_to_cpu in False True; do
        MISC_FAILED_KV_OFFLOAD_ARGS+=(
          "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-${_replicated}-cuda:0-0-4-256-64-${_blocks_per_chunk}-${_page_size}-3-${_gpu_to_cpu}]"
        )
      done
    done
  done
done

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-misc-failed-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [env] offline mode and runtime settings
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor compilation needs cuda.h + ptxas + libcuda; without
# them torch.compile tests fail with BackendCompilerFailed
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [diag] the expected root cause: numba absent from the nightly dependency set
# ------------------------------------------------------------------------------
python3 -c "import numba; print('[diag] numba', numba.__version__)" 2>/dev/null \
  || echo "[diag] numba NOT importable - collection of test_ngram.py will fail with ModuleNotFoundError"

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

root = ET.Element("testsuites", name="vLLM PPU Misc failed cases (GHA)")
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

lines = ["### Misc failed cases (PPU, v0.27.0)", "",
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
        try:
            agg = _stats(path)
        except (ET.ParseError, OSError):
            lines.append(f"| {label} | - | - | - | - | - | - | FAILED (bad junit) |")
            bad_units += 1
            continue
        n_t, n_f, n_e, n_s = (int(agg[k]) for k in COLS[:4])
        # a skip is not a verified pass for a targeted rerun
        ok = (n_f == 0 and n_e == 0 and n_s == 0)
        if not ok:
            bad_units += 1
        status = "PASSED" if ok else ("FAILED" if n_f or n_e else "SKIPPED")
        lines.append(f"| {label} | {n_t} | {n_t - n_f - n_e - n_s} | {n_f} "
                     f"| {n_e} | {n_s} | {agg['time']:.0f}s | "
                     f"{status} |")
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
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] ship the per-unit table back over NAS
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
# [run] step runner (single shard)
# ------------------------------------------------------------------------------
TOTAL_RC=0

_run_step() {
  local label="$1" shards="$2"
  shift 2
  local args=("$@")
  STEP_LABELS_LIST="${STEP_LABELS_LIST} ${label}"

  if [ "${shards}" -gt 1 ]; then
    echo "[step] ERROR: this script is single-shard only" >&2
    exit 2
  fi
  echo "========== [step] ${label} =========="
  local out_xml="${TMP_JUNIT}/${label}.xml"
  set +e
  pytest -v -s -rs "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} --junit-xml="${out_xml}"
  local rc=$?
  set -e
  if [ "${rc}" -eq 0 ] && [ -f "${out_xml}" ] && ! python3 - "${out_xml}" <<'PYEOF'
import sys
from xml.etree import ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
sys.exit(1 if sum(int(s.get("skipped") or 0) for s in root.iter("testsuite")) else 0)
PYEOF
  then
    echo "[step] ${label}: skipped cases are not a verified pass"
    rc=1
  fi
  echo "[step] ${label} rc=${rc}"
  if [ "${rc}" -ne 0 ]; then TOTAL_RC=1; fi
}

CUDA_VISIBLE_DEVICES=0 _run_step "misc_failed_ngram_collection" 1 "${MISC_FAILED_NGRAM_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "misc_failed_gpu_sync_debug" 1 "${MISC_FAILED_GPU_SYNC_DEBUG_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "misc_failed_kv_offload" 1 "${MISC_FAILED_KV_OFFLOAD_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
