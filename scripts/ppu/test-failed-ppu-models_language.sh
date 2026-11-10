#!/bin/bash
# ==============================================================================
# scripts/ppu/test-failed-ppu-models_language.sh -- rerun the v0.27.0
# models-language failure from nightly run 41978966338 (Nightly PPU test #18).
# Caller: .github/workflows/test_ppu_nightly.yml (in the container,
# cwd=/workspace/source, after ppu_ci_nightly_install_dependency.sh).
#
# Failing case (1 on v0.27.0):
#   tests/models/language/generation/test_hybrid.py::test_models[5-64-ibm-granite/granite-4.0-tiny-preview]
#
# The nightly ran it inside a 2-shard hybrid step and only the shard log tail
# reached the job log, so the assertion itself was never visible; every other
# hybrid model of the same step (Jamba, Zamba2, Falcon-H1, LFM2, qwen3-next)
# passed, and the case fails on v0.26.0 / v0.28.0 as well. The first rerun
# (PR #60) showed the vLLM engine refusing to start: the HF reference model
# (about 13 GiB, the largest of the hybrid list) is still resident on the
# device after HfRunner exits, so free memory (83/96 GiB) is below the
# 0.92 gpu_memory_utilization startup guard. This rerun localizes that:
#   1. hybrid_granite: the case as the nightly runs it;
#   2. hybrid_granite_mem_diag: only when step 1 fails, the same case with a
#      pytest plugin reporting the torch / driver memory views and the live
#      CUDA tensors after HfRunner exits, then polling up to 120 s for a lazy
#      driver release before vLLM starts. It is a diagnostic step and keeps
#      the area red whatever it reports.
# A skipped case does not count as verified: the step rc becomes 1 and the
# summary marks the unit SKIPPED.
#
# Modeled on scripts/ppu/test-area-ppu-models_language.sh of the
# ci/ppu-nightly-all-areas branch ([deps] hybrid probe, MODEL_MAP symlink,
# junit/summary emission); only the failing node ID runs here.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# workflow_dispatch pytest_args passthrough (kept for parity with the area script)
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

# ------------------------------------------------------------------------------
# [tests] exactly the case that failed in run 41978966338 on v0.27.0
# ------------------------------------------------------------------------------
ML_FAILED_GRANITE_ARGS=(
  "tests/models/language/generation/test_hybrid.py::test_models[5-64-ibm-granite/granite-4.0-tiny-preview]"
)

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-models-language-failed-junit"
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
# [deps] hybrid prerequisites (same probe order as the nightly area script:
# import -> install from the FlyTiger PyPI)
# ------------------------------------------------------------------------------
PIP_INSTALL="python3 -m pip install --no-cache-dir"
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
if ! python3 -c "import einops" 2>/dev/null; then
  echo "[deps] installing einops"
  ${PIP_INSTALL} einops -i "${PPU_PIP_INDEX}"
fi
HYBRID_READY=1
for pkg in mamba_ssm causal_conv1d; do
  if ! python3 -c "import ${pkg}" 2>/dev/null; then
    HYBRID_READY=0
    break
  fi
done
if [ "${HYBRID_READY}" = "0" ]; then
  echo "[deps] mamba_ssm/causal_conv1d not importable - trying flytiger PyPI"
  ${PIP_INSTALL} mamba-ssm causal-conv1d -i "${PPU_PIP_INDEX}" || true
fi
for pkg in mamba_ssm causal_conv1d; do
  python3 -c "import ${pkg}; print('[deps] ${pkg}', getattr(${pkg}, '__version__', '?'))" 2>/dev/null \
    || echo "[deps] ${pkg} NOT importable (the HF reference falls back to the torch path)"
done

# ------------------------------------------------------------------------------
# [setup] HF hub cache symlink: HF id -> local checkpoint under /nas_aisw
# ------------------------------------------------------------------------------
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    "ibm-granite/granite-4.0-tiny-preview": "/nas_aisw/datasets/checkpoints/LLM/granite/v1.0/granite-4.0-tiny-preview",
}

HF_CACHE = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
    "~/.cache/huggingface/hub")
os.makedirs(HF_CACHE, exist_ok=True)

for hf_id, local_path in MODEL_MAP.items():
    if not os.path.isdir(local_path):
        print(f"[setup] MISS  {hf_id} -> {local_path} (tests using it will fail)")
        continue
    org, repo = hf_id.split("/", 1)
    cache_dir = os.path.join(HF_CACHE, f"models--{org}--{repo}")
    snap_link = os.path.join(cache_dir, "snapshots", "main")
    refs_file = os.path.join(cache_dir, "refs", "main")
    if os.path.islink(snap_link) or os.path.exists(snap_link):
        print(f"[setup] KEEP  {hf_id} -> {os.path.realpath(snap_link)}")
        continue
    os.makedirs(os.path.dirname(snap_link), exist_ok=True)
    os.makedirs(os.path.dirname(refs_file), exist_ok=True)
    os.symlink(local_path, snap_link)
    with open(refs_file, "w") as rf:
        rf.write("main")
    print(f"[setup] OK    {hf_id} -> {local_path}")
PYEOF

# [diag] the stack the case depends on
python3 - <<'PYEOF' || true
import importlib.metadata as m
for pkg in ("torch", "transformers", "triton"):
    try:
        print(f"[diag] {pkg} {m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"[diag] {pkg} not installed")
PYEOF
echo "[diag] VLLM_PPU_MOE_BACKEND=${VLLM_PPU_MOE_BACKEND:-<unset, default priority>}"

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

root = ET.Element("testsuites", name="vLLM PPU Models Language failed cases (GHA)")
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

lines = ["### Models Language failed cases (PPU, v0.27.0)", "",
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

# [diag] pytest plugin: device memory right after HfRunner exits (torch view,
# driver view, live CUDA tensors and their referrers), then poll the driver
# view for up to 120 s before vLLM starts
DIAG_DIR=/tmp/ppu-models-language-diag
mkdir -p "${DIAG_DIR}"
cat > "${DIAG_DIR}/ppu_mem_diag.py" <<'PYEOF'
import gc
import sys
import time

import torch

_patched = False


def _gib(n):
    return f"{n / 2**30:.2f}"


def _report(tag):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(f"[mem-diag] {tag}: torch allocated={_gib(torch.cuda.memory_allocated())} "
          f"reserved={_gib(torch.cuda.memory_reserved())} "
          f"driver used={_gib(total - free)}/{_gib(total)} GiB", flush=True)
    return total - free


def _live_tensors():
    live = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                live.append(obj)
        except Exception:
            pass
    live.sort(key=lambda t: t.numel() * t.element_size(), reverse=True)
    total = sum(t.numel() * t.element_size() for t in live)
    print(f"[mem-diag] live CUDA tensors: {len(live)} ({_gib(total)} GiB)", flush=True)
    for t in live[:8]:
        refs = [type(r).__name__ for r in gc.get_referrers(t) if r is not live][:6]
        print(f"[mem-diag]   {tuple(t.shape)} {t.dtype} "
              f"{_gib(t.numel() * t.element_size())} GiB referrers={refs}", flush=True)
    live.clear()


def pytest_runtest_setup(item):
    global _patched
    if _patched:
        return
    for mod in list(sys.modules.values()):
        runner = getattr(mod, "HfRunner", None)
        if runner is None or not getattr(mod, "__name__", "").endswith("conftest"):
            continue
        orig_exit = runner.__exit__

        def _exit(self, *args, _orig=orig_exit):
            ret = _orig(self, *args)
            _report("after HfRunner exit")
            _live_tensors()
            gc.collect()
            torch.cuda.empty_cache()
            start = time.time()
            used = _report("after extra gc + empty_cache")
            while used > 4 * 2**30 and time.time() - start < 120:
                time.sleep(10)
                used = _report(f"poll +{time.time() - start:.0f}s")
            return ret

        runner.__exit__ = _exit
        _patched = True
        print(f"[mem-diag] patched {mod.__name__}.HfRunner.__exit__", flush=True)
        return
PYEOF

CUDA_VISIBLE_DEVICES=0 _run_step "hybrid_granite" 1 "${ML_FAILED_GRANITE_ARGS[@]}"
if [ "${TOTAL_RC}" -ne 0 ]; then
  echo "[diag] hybrid_granite failed - rerun with the memory diagnostics plugin"
  PYTHONPATH="${DIAG_DIR}${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES=0 \
    _run_step "hybrid_granite_mem_diag" 1 -p ppu_mem_diag "${ML_FAILED_GRANITE_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
