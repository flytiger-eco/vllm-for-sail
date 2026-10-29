#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-models_multimodal.sh -- PPU Models Multimodal tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single   - this area is single only (mirrors the core_model
#               subset of upstream Multi-Modal Models (Standard)); multimodal distributed is
#               covered by the models_distributed area, hence no multi section
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|all) ;;
# This area has no multi section (multimodal distributed is covered by the models_distributed
# area); it rejects the mode explicitly instead of passing silently, to avoid a false green CI.
multi) echo "[mode] ERROR: area models-multimodal has no multi section (multimodal distributed is covered by the models_distributed area)" >&2; exit 2 ;;
*) echo "[mode] ERROR: invalid TEST_MODE '${MODE}' (this area is single|all only)" >&2; exit 2 ;;
esac

# workflow_dispatch pytest_args passthrough: split on whitespace and append to
# the tail of every pytest command (e.g. `-k test_foo -x`) so a rerun can be narrowed without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-models-multimodal-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area-specific dependencies: opencv (cv2).
# ------------------------------------------------------------------------------
if python3 -c "import cv2" 2>/dev/null; then
  echo "[deps] opencv already importable: $(python3 -c 'import cv2; print(cv2.__version__)')"
else
  echo "[deps] installing opencv-python-headless from flytiger PyPI"
  python3 -m pip install --no-cache-dir opencv-python-headless \
    -i "https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple" || \
echo "[deps] WARN: opencv install failed - the test_common import chain crashes at collection"
fi

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/models_multimodal.yaml)
# ------------------------------------------------------------------------------
# Step mm_generation_dedicated - mirrors the non-test_common part of upstream Multi-Modal
# Models (Standard) 1-4: tests/models/multimodal -m core_model, excluding test_common and a
# batch of files blocked by the offline environment/dependencies. Among the non-ignored files
# only test_nemotron_parse.py (nvidia/NVIDIA-Nemotron-Parse-v1.1) and
# pooling/test_llava_next.py (royokong/e5-v, see MODEL_MAP) actually carry the core_model marker.
# The reason for every --ignore is kept verbatim from the original yaml:
MM_GENERATION_ARGS=(
  tests/models/multimodal
  -m
  core_model
# collection-time crash: test_common.py's module-level video_with_metadata_glm4_1v()
# -> hf_hub_download(sample_demo_1.mp4); an offline pod crashes during collection and -k/-m cannot avoid it
  "--ignore=tests/models/multimodal/generation/test_common.py"
# librosa missing (reference.md Landmine #17)
  "--ignore=tests/models/multimodal/generation/test_whisper.py"
  "--ignore=tests/models/multimodal/generation/test_phi4mm.py"
# mistral_common.audio dependency
  "--ignore=tests/models/multimodal/generation/test_voxtral.py"
  "--ignore=tests/models/multimodal/generation/test_voxtral_realtime.py"
# video assets are not vendored (the red zone has no internet)
  "--ignore=tests/models/multimodal/generation/test_qwen2_5_vl.py"
  "--ignore=tests/models/multimodal/generation/test_qwen2_vl.py"
  "--ignore=tests/models/multimodal/generation/test_interleaved.py"
# audio assets are downloaded from S3, the red zone has no internet
  "--ignore=tests/models/multimodal/generation/test_ultravox.py"
# trust_remote_code dynamic code download fails
  "--ignore=tests/models/multimodal/generation/test_keye.py"
# processing/ is handled separately by the mm_processing step
  "--ignore=tests/models/multimodal/processing"
# registry test that walks models without an alias
  "--ignore=tests/models/multimodal/test_mapping.py"
# the Prithvi models have no alias
  "--ignore=tests/models/multimodal/pooling/test_prithvi_mae.py"
# module-level read of the examples/ jinja templates, the CI tar.gz has no examples/
  "--ignore=tests/models/multimodal/pooling/test_llama_nemotron_vl.py"
# VLM2Vec-Full trust_remote_code download fails
  "--ignore=tests/models/multimodal/pooling/test_phi3v.py"
)

# Step mm_processing - mirrors upstream Multi-Modal Processor (Step 6): processor tensor
# schema validation. Note that test_tensor_schema.py starts with `from .test_common import
# get_model_ids_to_test, get_text_token_prompts`, so the import executes the test_common module
# body (including the module-level video download); this step therefore relies on opencv from
# [deps] and the offline fallback for sample_demo_1.mp4 from [assets]; missing either one
# crashes at collection (observed on the first run, see the aone yaml note about possibly
# triggering Landmine #19 (VllmConfig NoneType)).
MM_PROCESSING_ARGS=(
  tests/models/multimodal/processing/test_tensor_schema.py
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
# test_nemotron_parse.py::test_models (the only core_model, trust_remote_code=True)
# hit: aone_ci/scripts/ppu_model_aliases.json key nv-community/... ->
# ... (/ppusw -> /nas_aisw). The test uses the HF id nvidia/... (a different org pointing at
# the same weights), so the symlink is created under the id the test actually uses.
# trust_remote_code needs the dynamic modules (modeling_*.py etc.) inside the NAS directory;
# when missing the case fails (waiting for the .py files to be staged).
    "nvidia/NVIDIA-Nemotron-Parse-v1.1":
        "/nas_aisw/datasets/checkpoints/LLM/NVIDIA/v1.1/NVIDIA-Nemotron-Parse-v1.1",
# pooling/test_llava_next.py MODELS (core_model)
# hit: ppu_model_aliases.json royokong/e5-v ->
# /ppusw/datasets/checkpoints/LLM/e5/v1.0/e5-v (/ppusw -> /nas_aisw)
    "royokong/e5-v": "/nas_aisw/datasets/checkpoints/LLM/e5/v1.0/e5-v",
# Note: mm_processing (test_tensor_schema.py) walks the multimodal models of the
# HF_EXAMPLE_MODELS registry for processor validation and only needs config/tokenizer through
# dummy_hf_overrides; the set is large and most cases carry skip guards, so they are not
# enumerated here - watch the MISS report of the first run and add them as needed (same
# strategy as models-language).
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
# [assets] offline fallback for the vLLM public assets (images + video)
# ------------------------------------------------------------------------------
# The multimodal tests pull assets from s3://vllm-public-assets (images) and the HF dataset
# raushan-testing-hf/videos-test (video sample_demo_1.mp4) through vllm/assets, and the red
# zone has no internet, so they must be pre-staged. Two module-level touch points crash during
# collection:
#   - test_nemotron_parse.py starts with IMAGE = ImageAsset("paper-11")... -> needs the image
#   - test_common.py (imported by test_tensor_schema.py) downloads the video at module level -> needs the video
# Fallback mechanism (mirrors the aone_ci bootstrap, but with zero runtime dependency on
# aone_ci/, probing the /nas_aisw pre-staged volume instead):
#   - the image lands flat as $VLLM_ASSETS_CACHE/vllm_public_assets/<file>
#     (see vllm/assets/base.py:get_vllm_public_assets, a hit skips the download)
#   - the video lands flat as $VLLM_ASSETS_CACHE/video-example-data/sample_demo_1.mp4
#     (see vllm/assets/video.py:download_video_asset, an existing video_path skips the download)
# When missing it warns and marks the asset as pending staging instead of guessing a NAS path;
# the first run adds the measured candidate paths.
echo "========== [assets] bootstrap vLLM public assets (/nas_aisw) =========="
python3 - <<'PYEOF'
import os
import shutil

CACHE = os.environ.get("VLLM_ASSETS_CACHE") or os.path.expanduser(
    "~/.cache/vllm/assets")
IMG_DST = os.path.join(CACHE, "vllm_public_assets")
VID_DST = os.path.join(CACHE, "video-example-data")

# candidate directories for the vLLM public assets on /nas_aisw (narrowed by the first run; the red-zone list does not track such assets)
IMG_SRC_CANDS = [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/vision_model_images",
    "/nas_aisw/datasets/assets/vllm_public_assets",
]
VID_SRC_CANDS = [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/multimodal_asset",
    "/nas_aisw/datasets/assets/vllm_public_assets",
]

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
if not os.path.exists(os.path.join(IMG_DST, "paper-11.png")):
    print("[assets] WARN: paper-11.png absent (not staged) - "
          "test_nemotron_parse collection will crash")
if not os.path.exists(os.path.join(VID_DST, "sample_demo_1.mp4")):
    print("[assets] WARN: sample_demo_1.mp4 absent (not staged) - "
          "test_common/test_tensor_schema collection will crash")
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

root = ET.Element("testsuites", name="vLLM PPU Models Multimodal (GHA)")
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

lines = ["### Models Multimodal Test (PPU)", "",
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

# Mirrors the aone single section: both single and all run the mm_generation_dedicated +
# mm_processing steps (this area has no multi section, the mode is rejected above). The Aone
# pod is 1-PPU, so the visible cards are limited to 1 (CUDA_VISIBLE_DEVICES=0) to keep single-card semantics.
CUDA_VISIBLE_DEVICES=0 _run_step "mm_generation_dedicated" 1 "${MM_GENERATION_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "mm_processing" 1 "${MM_PROCESSING_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
