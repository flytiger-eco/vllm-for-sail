#!/bin/bash
# ==============================================================================
# scripts/ppu/test-failed-ppu-entrypoints.sh -- rerun the v0.28.0 entrypoints
# cases that failed in nightly run 41978966338 (Nightly PPU test #18).
# Caller: .github/workflows/test_ppu_nightly.yml (in the container,
# cwd=/workspace/source, after ppu_ci_nightly_install_dependency.sh).
#
# Failing cases (4 on v0.28.0):
#   tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_documents_field_accepted
#   tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_non_streaming
#   tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_streaming
#   tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_validation_error_returns_400
#
# Root cause: every request got 404. The server log says
# "VLLM_ENABLE_COHERE_API=1 but the `cohere` SDK is not installed":
# vllm.entrypoints.cohere.api_router.attach_router only registers
# POST /cohere/v2/chat when `import cohere` succeeds, and the cohere SDK is a
# test-only dependency (requirements/test/cuda.txt) that neither
# requirements/ppu.txt nor the image ships. The same missing SDK is why the
# nightly ignores test_api_router.py / test_protocol.py /
# test_serving_conversion.py / test_serving_streaming.py on v0.28.0 (collection
# error), so those files are rerun as a second step.
# Fix under verification: [deps] installs the versions pinned by
# requirements/test/cuda.txt (cohere==7.0.8, cohere-melody==0.11.1) under a
# constraint on the preinstalled packages, so the image stack is not moved.
# A skipped case does not count as verified: the step rc becomes 1 and the
# summary marks the unit SKIPPED.
#
# Modeled on scripts/ppu/test-area-ppu-entrypoints.sh of the
# ci/ppu-nightly-all-areas branch (env blocks, junit/summary emission); only
# the cohere cases run here.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# workflow_dispatch pytest_args passthrough (kept for parity with the area script)
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

# ------------------------------------------------------------------------------
# [tests] exactly the cases that failed in run 41978966338 on v0.28.0
# ------------------------------------------------------------------------------
EP_FAILED_COHERE_CHAT_ARGS=(
  "tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_documents_field_accepted"
  "tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_non_streaming"
  "tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_streaming"
  "tests/entrypoints/cohere/test_chat_v2.py::test_cohere_v2_chat_validation_error_returns_400"
)

# the cohere files the nightly ignores on v0.28.0 for the same missing SDK
EP_COHERE_SDK_ARGS=(
  tests/entrypoints/cohere/test_api_router.py
  tests/entrypoints/cohere/test_protocol.py
  tests/entrypoints/cohere/test_serving_conversion.py
  tests/entrypoints/cohere/test_serving_streaming.py
)

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-entrypoints-failed-junit"
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
# [deps] the optional cohere SDK (test-only dependency, pinned by
# requirements/test/cuda.txt). Constrain every preinstalled distribution to its
# current version so the SDK cannot move the image stack (torch, tokenizers,
# pydantic, httpx, ...); an unsatisfiable pin fails here instead of later.
# Artifactory first, then the aliyun mirror.
# ------------------------------------------------------------------------------
PIP_INSTALL="python3 -m pip install --no-cache-dir --retries 5 --timeout 30"
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
COHERE_REQS=("cohere==7.0.8" "cohere-melody==0.11.1")
COHERE_CONSTRAINTS="/tmp/ppu-cohere-constraints.txt"
python3 -m pip list --format=freeze 2>/dev/null \
  | grep -viE '^(cohere|cohere-melody|vllm)==' > "${COHERE_CONSTRAINTS}" || true
if python3 -c "import cohere, cohere_melody" 2>/dev/null; then
  echo "[deps] cohere SDK already importable, keep it"
elif ${PIP_INSTALL} "${COHERE_REQS[@]}" -c "${COHERE_CONSTRAINTS}" -i "${PPU_PIP_INDEX}" \
    || ${PIP_INSTALL} "${COHERE_REQS[@]}" -c "${COHERE_CONSTRAINTS}" -i "https://mirrors.aliyun.com/pypi/simple/"; then
  echo "[deps] ${COHERE_REQS[*]} installed"
else
  echo "[deps] WARNING: cohere SDK install failed; the cohere routes stay unregistered (404)" >&2
fi
python3 -c "import cohere, cohere_melody; print(f'[verify] cohere {cohere.__version__}, cohere_melody importable')" \
  || echo "[verify] cohere SDK NOT importable - test_chat_v2 will keep returning 404" >&2

# ------------------------------------------------------------------------------
# [junit] merge helper (with a fallback when pytest crashes)
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329 # invoked through `trap _emit_junit EXIT`
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

root = ET.Element("testsuites", name="vLLM PPU Entrypoints failed cases (GHA)")
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

lines = ["### Entrypoints failed cases (PPU, v0.28.0)", "",
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
# [run] step runner (single-shard only: the failing subset is small)
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

CUDA_VISIBLE_DEVICES=0 _run_step "cohere_chat_v2" 1 "${EP_FAILED_COHERE_CHAT_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "cohere_sdk_files" 1 "${EP_COHERE_SDK_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
