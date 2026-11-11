#!/bin/bash
# ==============================================================================
# scripts/ppu/test-failed-ppu-models_multimodal.sh -- focused v0.28.0
# MiniCPM-o processing canary for the torchaudio 2.11 compatibility experiment.
# Caller: .github/workflows/test_ppu_nightly.yml (in the container,
# cwd=/workspace/source, after ppu_ci_nightly_install_dependency.sh).
#
# Only test_model_tensor_schema[openbmb/MiniCPM-o-2_6] runs. The dependency
# script tries the PPU torchaudio 2.11.0 build first, then aliyun 2.11.1, with
# --no-deps so the image torch is not replaced. deps-runtime.txt records the
# actual imports; a skipped case does not count as a verified pass.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# workflow_dispatch pytest_args passthrough (kept for parity with the area script)
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

MM_TORCHAUDIO_CANARY_ARGS=(
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[openbmb/MiniCPM-o-2_6]"
)

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-models-multimodal-failed-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [env] offline mode and runtime settings
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export PYTHONUNBUFFERED=1

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

PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [deps] snapshot the environment the case runs in. The vllm wheel is installed
# with --target into the source tree, so deps-runtime.txt also records the import
# paths rather than relying on pip metadata alone.
# ------------------------------------------------------------------------------
echo "========== [deps] snapshot -> ${RESULTS_DIR}/deps-*.txt =========="
python3 -m pip freeze --all > "${RESULTS_DIR}/deps-pip-freeze.txt" 2>&1 || true
python3 -m pip list -v > "${RESULTS_DIR}/deps-pip-list.txt" 2>&1 || true
python3 -m pip check > "${RESULTS_DIR}/deps-pip-check.txt" 2>&1 || true
python3 - > "${RESULTS_DIR}/deps-runtime.txt" 2>&1 <<'PYEOF' || true
import importlib
import sys

print(f"python\t{sys.version.split()[0]}\t{sys.executable}")
for name in ("vllm", "torch", "torchaudio", "torchvision", "transformers",
             "huggingface_hub", "tokenizers", "numpy", "triton"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}\t{getattr(mod, '__version__', '?')}\t{getattr(mod, '__file__', '?')}")
    except Exception as e:  # noqa: BLE001 - record, do not fail the run
        print(f"{name}\tNOT IMPORTABLE\t{type(e).__name__}: {e}")
PYEOF
cat "${RESULTS_DIR}/deps-runtime.txt"

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

root = ET.Element("testsuites", name="vLLM PPU Models Multimodal failed cases (GHA)")
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

lines = ["### Models Multimodal failed cases (PPU, v0.28.0)", "",
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
  # [failed-cases] one row per failed case with its root error, taken from the
  # junit traceback (the forked runner wraps it in "Test ... failed when called")
  python3 - "${RESULTS_DIR}/test.xml" "${RESULTS_DIR}/failed-cases.tsv" <<'PYEOF' || true
import re
import sys
from xml.etree import ElementTree as ET

src, out = sys.argv[1:]
rows = []
for ts in ET.parse(src).getroot().iter("testsuite"):
    for tc in ts.iter("testcase"):
        kind, bad = "FAILED", tc.find("failure")
        if bad is None:
            kind, bad = "ERROR", tc.find("error")
        if bad is None:
            continue
        elines = [ln for ln in (bad.text or "").splitlines() if re.match(r"^E\s", ln)]
        elines = [re.sub(r"^E\s+", "", ln).strip() for ln in elines]
        cause = (bad.get("message") or "").strip().splitlines()[:1]
        for i, ln in enumerate(elines):
            if re.search(r"(Error|Exception)\b", ln) and not ln.startswith("AssertionError: Test "):
                cause = elines[i:i + 3]
                break
        cls = tc.get("classname") or ""
        nodeid = f"{cls.replace('.', '/')}.py::{tc.get('name')}" if cls else tc.get("name")
        rows.append("\t".join((nodeid, kind, ts.get("name") or "",
                               " | ".join(cause).replace("\t", " ")[:500])))
with open(out, "w") as f:
    f.write("nodeid\tstatus\tunit\tcause\n")
    f.writelines(r + "\n" for r in rows)
print(f"[failed-cases] {len(rows)} rows -> {out}")
PYEOF
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-results] ship all of test-results/ back over NAS for the step summary
  # and the test-results-* artifact, including when the canary fails.
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -rf "${RESULTS_DIR}"/. "${PPU_SUMMARY_NAS_DIR}/"; then
      chmod -R a+rX "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null || true
      echo "[k8s-results] $(ls "${RESULTS_DIR}" | tr '\n' ' ')-> ${PPU_SUMMARY_NAS_DIR}/"
    else
      echo "[k8s-results] WARN: NAS upload failed (is ${PPU_SUMMARY_NAS_DIR} writable?)"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] step runner (single shard; the pod is 1-PPU, CUDA_VISIBLE_DEVICES=0)
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
  pytest -v -s -rs "${args[@]}" ${PYTEST_EXTRA[@]+"${PYTEST_EXTRA[@]}"} --junit-xml="${out_xml}" \
    2>&1 | tee "${TMP_JUNIT}/${label}.log"
  local rc=${PIPESTATUS[0]}
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

CUDA_VISIBLE_DEVICES=0 _run_step "mm_failed_processing" 1 "${MM_TORCHAUDIO_CANARY_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
