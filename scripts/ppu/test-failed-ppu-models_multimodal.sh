#!/bin/bash
# ==============================================================================
# scripts/ppu/test-failed-ppu-models_multimodal.sh -- rerun the v0.26.0
# models-multimodal cases that failed in nightly run 41978966338
# (Nightly PPU test #18).
# Caller: .github/workflows/test_ppu_nightly.yml (in the container,
# cwd=/workspace/source, after ppu_ci_nightly_install_dependency.sh).
#
# Failing cases:
#  - mm_generation_dedicated: 10 cases (first failing run 41379000962, still
#    failing in 41978966338; CI keeps them deselected for v0.26/v0.27):
#      tests/models/multimodal/generation/test_mm_prefix_lm.py::test_mm_prefix_lm_e2e
#      tests/models/multimodal/generation/test_moss_audio.py::test_moss_audio_generation_smoke
#      tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[qwen3_vl]
#      tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[internvl]
#      tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[{qwen2_vl,qwen2_5_vl,qwen3_vl,qwen3_5,internvl,glm4_1v}]
#  - mm_processing: every test_tensor_schema case failed (108/108 on v0.26.0,
#    110/110 on v0.27.0), so the whole file is rerun here.
#
# Root causes (run 42200535187) and fixes under verification:
#  - mm_prefix_lm: the PPU flash_attn_interface asserted on the FA4 capability
#    probe; it now reports unsupported versions (source fix).
#  - vit_cudagraph_video: sample_demo_1.mp4 was never staged; [assets] now also
#    takes it from the HF cache dataset raushan-testing-hf/videos-test.
#  - missing repos (LocalEntryNotFoundError / Invalid repository ID) and
#    non-hash snapshots (PosixPath / None in transformers dynamic modules):
#    ppu_mm_hf_cache.py links /nas_aisw checkpoints with a 40-hex revision
#    ([setup]) and lists what the offline cache still lacks ([preflight],
#    test-results/mm-hf-cache-missing.tsv); fill it on a networked host with
#    `ppu_mm_hf_cache.py prefetch --list <tsv>`.
#  - MiniCPM-o: torchaudio is now installed by ppu_ci_nightly_install_dependency.sh.
# A skipped case does not count as verified: the step rc becomes 1 and the
# summary marks the unit SKIPPED.
#
# Modeled on scripts/ppu/test-area-ppu-models_multimodal.sh of the
# ci/ppu-nightly-all-areas branch (env, MODEL_MAP symlinks, assets bootstrap,
# junit/summary emission); the area-wide -m core_model selection is dropped
# because this script runs only the failing node IDs plus the failing file.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# workflow_dispatch pytest_args passthrough (kept for parity with the area script)
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

# ------------------------------------------------------------------------------
# [tests] exactly the cases that failed in run 41978966338 on v0.26.0
# ------------------------------------------------------------------------------
# Generation failures; kept in one pytest step so one shared fixture/import
# problem shows up as a single step failure, not 10 separate rows.
MM_FAILED_GENERATION_ARGS=(
  "tests/models/multimodal/generation/test_mm_prefix_lm.py::test_mm_prefix_lm_e2e"
  "tests/models/multimodal/generation/test_moss_audio.py::test_moss_audio_generation_smoke"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[qwen3_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_image[internvl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen2_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen2_5_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen3_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen3_5]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[internvl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[glm4_1v]"
)

# Processing failures: 108/108 test_tensor_schema cases failed in run
# 41978966338 on v0.26.0, so rerun the whole file.
MM_FAILED_PROCESSING_ARGS=(
  tests/models/multimodal/processing/test_tensor_schema.py
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
# [setup] HF hub cache symlink: HF id -> local checkpoint under /nas_aisw
# (MODEL_MAP plus a by-name probe for the generation models; extend with
# PPU_MM_MODEL_MAP="org/a=/path/a,..."). The snapshot is named by a 40-hex
# revision because trust_remote_code models cannot load from snapshots/main.
# ------------------------------------------------------------------------------
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 scripts/ppu/ppu_mm_hf_cache.py link \
  || echo "[setup] WARN: HF cache link step failed; generation cases may miss checkpoints"

# ------------------------------------------------------------------------------
# [assets] offline fallback for the vLLM public assets (images + video)
# download_video_asset() returns <VLLM_ASSETS_CACHE>/video-example-data/<file>
# as-is when present and only otherwise calls hf_hub_download, which fails in
# offline mode (vit_cudagraph_video: OfflineModeIsEnabled).
# ------------------------------------------------------------------------------
echo "========== [assets] bootstrap vLLM public assets (/nas_aisw) =========="
python3 - <<'PYEOF'
import glob
import os
import shutil

CACHE = os.environ.get("VLLM_ASSETS_CACHE") or os.path.expanduser(
    "~/.cache/vllm/assets")
IMG_DST = os.path.join(CACHE, "vllm_public_assets")
VID_DST = os.path.join(CACHE, "video-example-data")

IMG_SRC_CANDS = [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/vision_model_images",
    "/nas_aisw/datasets/assets/vllm_public_assets",
]
VID_SRC_CANDS = [
    p for p in (os.environ.get("PPU_VIDEO_ASSET_SRC"),) if p
] + [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/multimodal_asset",
    "/nas_aisw/datasets/assets/vllm_public_assets",
] + sorted(glob.glob(os.path.join(
    os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"),
    "datasets--raushan-testing-hf--videos-test", "snapshots", "*")))

def _bootstrap(dst, cands, only=None):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for src in cands:
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if not os.path.isfile(sp):
                continue
            if only is not None and f not in only:
                continue
            dp = os.path.join(dst, f)
            if not os.path.exists(dp):
                shutil.copy(sp, dp)
                n += 1
    return n

img_n = _bootstrap(IMG_DST, IMG_SRC_CANDS)
vid_n = _bootstrap(VID_DST, VID_SRC_CANDS, only={"sample_demo_1.mp4"})
print(f"[assets] images bootstrapped={img_n} -> {IMG_DST}")
print(f"[assets] video bootstrapped={vid_n} -> {VID_DST}")
if not os.path.exists(os.path.join(VID_DST, "sample_demo_1.mp4")):
    print("[assets] WARN: sample_demo_1.mp4 absent (not staged) - "
          "test_vit_cudagraph_video cases will fail")
PYEOF

# ------------------------------------------------------------------------------
# [preflight] diagnostic only: list what the offline HF cache cannot serve, so a
# red step points at the cache gap instead of 100 near-identical tracebacks.
# ------------------------------------------------------------------------------
echo "========== [preflight] offline HF cache coverage =========="
python3 scripts/ppu/ppu_mm_hf_cache.py check --out "${RESULTS_DIR}/mm-hf-cache-missing.tsv" \
  || echo "[preflight] WARN: HF cache coverage check failed to run"

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

lines = ["### Models Multimodal failed cases (PPU, v0.26.0)", "",
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
      # the [preflight] gap list is the input of `ppu_mm_hf_cache.py prefetch --list`
      cp -f "${RESULTS_DIR}/mm-hf-cache-missing.tsv" "${PPU_SUMMARY_NAS_DIR}/" 2>/dev/null || true
    else
      echo "[k8s-summary] WARN: NAS upload failed (is ${PPU_SUMMARY_NAS_DIR} writable?)"
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

CUDA_VISIBLE_DEVICES=0 _run_step "mm_failed_generation" 1 "${MM_FAILED_GENERATION_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "mm_failed_processing" 1 "${MM_FAILED_PROCESSING_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
