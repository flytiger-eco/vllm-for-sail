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
# The arrays below retain the failures from PR #57 run 42660076007 as the
# baseline for a later broader rerun. This run executes only MiniCPM-o to test
# whether torchaudio 2.11 can coexist with the image's torch 2.13 build.
# Everything under test-results/ (junit, summary, per-step pytest logs,
# failed-cases.tsv with the root error per case, deps-*.txt dependency
# snapshot, mm-hf-cache-missing.tsv) is shipped to PPU_SUMMARY_NAS_DIR and
# uploaded by the workflow as the test-results-* artifact.
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
# [tests] historical failed-case baseline; only the canary below is executed
# ------------------------------------------------------------------------------
# Generation failures from run 42660076007, retained for a later broader rerun.
MM_FAILED_GENERATION_ARGS=(
  "tests/models/multimodal/generation/test_moss_audio.py::test_moss_audio_generation_smoke"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen2_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen2_5_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen3_vl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[qwen3_5]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[internvl]"
  "tests/models/multimodal/generation/test_vit_cudagraph.py::test_vit_cudagraph_video[glm4_1v]"
)

# Processing failures: 105 of the 153 test_tensor_schema cases still failed in
# run 42660076007 (48 passed). Keep this baseline out of the current canary run.
MM_FAILED_PROCESSING_ARGS=(
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[moondream/moondream3-preview0]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[CohereLabs/cohere-transcribe-03-2026]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[moonshotai/Kimi-VL-A3B-Thinking]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[fixie-ai/ultravox-v0_5-llama-3_2-1b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[kakaocorp/kanana-1.5-v-3b-instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[openvla/openvla-7b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[zai-org/GLM-4.5V]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16_0]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/audio-flamingo-3-hf]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Qwen/Qwen3-ASR-0.6B0]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[llava-hf/LLaVA-NeXT-Video-7B-hf]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[openbmb/MiniCPM-V-4_6]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Mistral-Small-3.1-24B-Instruct-2503]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Voxtral-Mini-4B-Realtime-2602]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[allenai/Molmo2-8B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[allenai/Molmo2-O-7B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Voxtral-Mini-3B-2507]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Kwai-Keye/Keye-VL-1_5-8B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ibm-granite/granite-speech-4.1-2b-plus]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[zai-org/GLM-4.6V-Flash]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[moondream/moondream3-preview1]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[deepseek-ai/DeepSeek-OCR-2]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Eagle2.5-8B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL3-1B-hf]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL2-1B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL2-2B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL3_5-1B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL3_5-30B-A3B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mispeech/midashenglm-7b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[deepseek-ai/deepseek-vl2-tiny]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Isotr0py/deepseek-vl2-tiny]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[zai-org/glm-4v-9b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[baidu/ERNIE-4.5-VL-28B-A3B-PT]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Kwai-Keye/Keye-VL-8B-Preview]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/NVIDIA-Nemotron-Parse-v1.2]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[LGAI-EXAONE/EXAONE-4.5-33B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Salesforce/blip2-opt-2.7b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Salesforce/blip2-opt-6.7b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[LiquidAI/LFM2-VL-450M]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Cosmos3-Nano]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Cosmos3-Super]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Qwen/Qwen2.5-Omni-7B-AWQ]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[rhymes-ai/Aria]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ai9stars/Cheers0]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ai9stars/Cheers1]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ibm-granite/granite-vision-3.3-2b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[PaddlePaddle/PaddleOCR-VL]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[thinkingmachines/Inkling-NVFP4]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[HuggingFaceTB/SmolVLM2-2.2B-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[baidu/Unlimited-OCR]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ibm-granite/granite-speech-3.3-2b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ibm-granite/granite-4.0-1b-speech]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[tencent/HunyuanOCR]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[stepfun-ai/step3]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[jinaai/jina-reranker-m0]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Pixtral-12B-2409]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[mistralai/Ministral-3-3B-Instruct-2512]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[ByteDance-Seed/BAGEL-7B-MoT]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Qwen/Qwen3-ASR-0.6B1]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Skywork/Skywork-R1V-38B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[microsoft/Phi-3-vision-128k-instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[microsoft/Phi-3.5-vision-instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[CohereLabs/command-a-vision-07-2025]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[CohereLabs/command-a-plus-05-2026-bf16]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/NVLM-D-72B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[zai-org/GLM-OCR]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[AIDC-AI/Ovis2-1B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[AIDC-AI/Ovis1.6-Llama3.2-3B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[AIDC-AI/Ovis1.6-Gemma2-9B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[baidu/Qianfan-OCR]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[zai-org/GLM-ASR-Nano-2512]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenMOSS-Team/MOSS-Audio-4B-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenMOSS-Team/MOSS-Audio-4B-Thinking]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenMOSS-Team/MOSS-Audio-8B-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[OpenMOSS-Team/MOSS-Audio-8B-Thinking]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Open-Bee/Bee-8B-RL]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[google/paligemma-3b-mix-224]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[google/paligemma2-3b-ft-docci-448]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[YannQi/R-4B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[stepfun-ai/Step-3.7-Flash]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[google/gemma-3n-E2B-it]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[meta-llama/Llama-Guard-4-12B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[openai/whisper-large-v3-turbo]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[naver-hyperclovax/HyperCLOVAX-SEED-Think-32B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[google/diffusiongemma-26B-A4B-it]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[h2oai/h2ovl-mississippi-800m]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[h2oai/h2ovl-mississippi-2b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16_1]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[openbmb/MiniCPM-o-2_6]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[rednote-hilab/dots.ocr]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[facebook/chameleon-7b]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[moonshotai/Kimi-Audio-7B-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[internlm/Intern-S1-Pro]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[microsoft/Phi-4-multimodal-instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[HuggingFaceTB/SmolVLM-256M-Instruct]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[lightonai/LightOnOCR-1B-1025]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Qwen/Qwen2.5-Omni-3B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[Qwen/Qwen3-ForcedAligner-0.6B]"
  "tests/models/multimodal/processing/test_tensor_schema.py::test_model_tensor_schema[stepfun-ai/Step3-VL-10B]"
)

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
# pytest output is tee'd into a per-step log; keep it line-buffered
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
# [deps] dependency snapshot of the environment the cases run in, shipped with
# the results so a failing case can be checked against what was installed.
# The vllm wheel is installed with --target into the source tree and does not
# appear in pip's view; deps-runtime.txt records what each module resolves to.
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
  # [k8s-results] ship all of test-results/ back over NAS: summary.md feeds the
  # step summary, the rest is uploaded as the test-results-* artifact (the
  # [preflight] gap list is the input of `ppu_mm_hf_cache.py prefetch --list`)
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
