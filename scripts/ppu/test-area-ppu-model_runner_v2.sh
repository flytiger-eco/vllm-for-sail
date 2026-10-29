#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-model_runner_v2.sh -- PPU Model Runner V2 tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single | multi
#
# Key: every step of this area needs VLLM_USE_V2_MODEL_RUNNER=1 (upstream buildkite
# model_runner_v2.yaml exports it per step and it is the core switch of this area) - see [env].
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch pytest_args passthrough: split on whitespace and append to
# the tail of every pytest command (e.g. `-k test_foo -x`) so a rerun can be narrowed without editing this script.
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-model-runner-v2-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area-specific pip dependencies: none.
# ------------------------------------------------------------------------------
# The upstream buildkite model_runner_v2.yaml Examples step runs `pip install tensorizer`,
# but the PPU scope of this area has no Examples (python3 scripts, not pytest, and many MM
# models are not staged; see the audit summary of aone_ci/ppu_extras/model_runner_v2.yaml).
# The core/distributed/spec_decode steps only use the image stack plus the pytest toolchain
# already installed by ppu_install_dependency.sh (pytest-asyncio/tblib/pytest-shard/pyyaml).

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single/multi sections in aone_ci/ppu_extras/model_runner_v2.yaml)
# ------------------------------------------------------------------------------
# single = the PPU version of the upstream buildkite "Model Runner V2 Core Tests" + "Spec
# Decode" steps (single card), mapped back to the single section of ppu_extras model_runner_v2.yaml.

# mrv2_core_llm_engine: v1/engine/test_llm_engine.py (opt-125m).
# test_engine_metrics is excluded (upstream buildkite does the same; PPU has no metrics deps).
MRV2_CORE_LLM_ENGINE_ARGS=(
  tests/v1/engine/test_llm_engine.py
  -k
  "not test_engine_metrics"
)

# mrv2_core_async_scheduling: v1/e2e/general/test_async_scheduling.py.
# Upstream buildkite uses `not ngram`; the PPU extras append `and not eagle3`
# (the eagle3 speculator fails the model redirect path rewrite assertion on PPU,
# LANDMINE #18). The whole step needs ENFORCE_EAGER=1 (see the prefix in the mode dispatch;
# upstream note: eager is required until the CG correctness issue is solved, removable after PR #32936).
MRV2_CORE_ASYNC_SCHEDULING_ARGS=(
  tests/v1/e2e/general/test_async_scheduling.py
  -k
  "not ngram and not eagle3"
)

# mrv2_core_context_length: v1/e2e/general/test_context_length.py (JackFram/llama-160m)
MRV2_CORE_CONTEXT_LENGTH_ARGS=(
  tests/v1/e2e/general/test_context_length.py
)

# mrv2_core_min_tokens: v1/e2e/general/test_min_tokens.py (opt-125m)
MRV2_CORE_MIN_TOKENS_ARGS=(
  tests/v1/e2e/general/test_min_tokens.py
)

# mrv2_core_struct_output: entrypoints/llm/test_struct_output_generate.py
# (since v0.23 the whole tests/v1/entrypoints/ tree moved into tests/entrypoints/).
# Only the xgrammar backend runs (upstream buildkite does the same); the 4 speculative_config
# parametrizations (ngram spec decoding) are excluded, they do not run on PPU.
MRV2_CORE_STRUCT_OUTPUT_ARGS=(
  tests/entrypoints/llm/test_struct_output_generate.py
  -k
  "xgrammar and not speculative_config6 and not speculative_config7 and not speculative_config8 and not speculative_config0"
)

# mrv2_spec_decode_e2e: the eagle/ + speculators/ trees after the v0.27 split.
# The old test_spec_decode.py and test_lora_with_spec_decode.py moved away and the lora
# combination now lives in draft_model/, which this step does not collect, so no extra ignore.
MRV2_SPEC_DECODE_E2E_ARGS=(
  tests/v1/e2e/spec_decode/eagle
  tests/v1/e2e/spec_decode/speculators
  -k
  "not qwen3_eagle3_speculator and not llama3_eagle3_speculator"
)
if [[ "${PPU_RELEASE:-}" == "v0.26.0" ]]; then
  MRV2_SPEC_DECODE_E2E_ARGS=(
    tests/v1/e2e/spec_decode/test_spec_decode.py
    -k
    "eagle or speculator"
  )
elif [[ "${PPU_RELEASE:-}" != "v0.27.0" && "${PPU_RELEASE:-}" != "v0.28.0" ]]; then
  echo "[tests] ERROR: PPU_RELEASE must identify a supported release" >&2
  exit 2
fi

# multi = the PPU version of the upstream buildkite "Distributed (2 GPUs)" + "Pipeline
# Parallelism" steps (the tests use 2 cards internally); mapped back to the multi section.

# mrv2_distributed_basic_correctness: the distributed(num_gpus=2) group of
# basic_correctness/test_basic_correctness.py. `not ray` drops the ray backend (hangs on PPU);
# `not True` drops the prompt_embeds parametrizations (unsupported on PPU; the upstream note
# calls the filter hacky as well). Needs TARGET_TEST_SUITE=L4 (see the mode dispatch prefix).
MRV2_DISTRIBUTED_BASIC_CORRECTNESS_ARGS=(
  tests/basic_correctness/test_basic_correctness.py
  -m
  "distributed(num_gpus=2)"
  -k
  "not ray and not True"
)

# mrv2_distributed_dp: v1/distributed/test_async_llm_dp.py with DP_SIZE=2.
# `not ray` (ray DP hangs); `not tiny` (tiny-random related); `not (dp_pause and False)`:
# the dp_pause case id carries no model name (parametrized over expert_parallel=[False,True]),
# the [False] branch hard-codes tiny-random(head_dim=4) and hits the FlexAttention inductor
# compile (needs E>=16), so the [True] (MoE+EP) branch is kept. Needs TP_SIZE=1 DP_SIZE=2
# NCCL_CUMEM_HOST_ENABLE=0 (see the prefix).
MRV2_DISTRIBUTED_DP_ARGS=(
  tests/v1/distributed/test_async_llm_dp.py
  -k
  "not ray and not tiny and not (dp_pause and False)"
)

# mrv2_distributed_eagle_dp: v1/distributed/test_eagle_dp.py with DP_SIZE=2.
# Needs TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0 (see the prefix).
MRV2_DISTRIBUTED_EAGLE_DP_ARGS=(
  tests/v1/distributed/test_eagle_dp.py
)

# mrv2_pipeline_parallelism: distributed/test_pipeline_parallel.py PP.
# Upstream buildkite uses `not ray and not Jamba`; the PPU extras append `not PowerLM and
# not DeepSeek` (the PowerLM-3b / DeepSeek-V2-Lite architectures crash engine init on PPU,
# LANDMINE #20). Note: create_new_process_for_each_test already goes through spawn on PPU
# (the is_ppu() branch of tests/utils.py), so there is no fork device-init landmine. The
# upstream 4-GPU PP (Llama-4-Scout) needs a 4-PPU pod and is not part of this 2-card area.
MRV2_PIPELINE_PARALLELISM_ARGS=(
  tests/distributed/test_pipeline_parallel.py
  -k
  "not ray and not Jamba and not PowerLM and not DeepSeek"
# Run 34936679139 (8 failed / 8 passed): Ilama-3.2-1B setup13 plus Phi-3.5-MoE
# setup14/16/18/20/22 - 6 cases with RuntimeError: Server exited unexpectedly (the engine
# process crashes under PP); InternVL2-1B setup0 and ultravox setup2 - 2 cases fail to load
# the multimodal model (the whisper-large-v3-turbo feature extractor is missing from the NAS
# cache, offline OSError). The other 8 cases in the file pass and stay.
# Restore: drop these one by one once PP is fixed / the whisper family is staged.
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[hmellor/Ilama-3.2-1B-parallel_setup13-mp-auto-test_options13]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup14-mp-auto-test_options14]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup16-mp-auto-test_options16]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup18-mp-auto-test_options18]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup20-mp-auto-test_options20]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_language_generation[microsoft/Phi-3.5-MoE-instruct-parallel_setup22-mp-auto-test_options22]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_multimodal_generation[OpenGVLab/InternVL2-1B-parallel_setup0-mp-auto-test_options0]"
  --deselect "tests/distributed/test_pipeline_parallel.py::test_tp_multimodal_generation[fixie-ai/ultravox-v0_5-llama-3_2-1b-parallel_setup2-mp-auto-test_options2]"
)

# ------------------------------------------------------------------------------
# [env] offline mode and runtime settings
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# Core switch of this area: enable the V2 model runner (the upstream buildkite file exports
# VLLM_USE_V2_MODEL_RUNNER=1 for every step and ppu_extras declares it as an area-level env.
# The auto-generated aone script missed this line; the GHA version restores it, otherwise the
# whole area loses its meaning).
export VLLM_USE_V2_MODEL_RUNNER="1"

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
    # ---- single: core tests ----
    # opt-125m: test_llm_engine.py / test_min_tokens.py / basic_correctness distributed
# list hit: checkpoints_cleaned.json path=checkpoints/LLM/misc/v1.0/opt-125m
    "facebook/opt-125m": "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
    # test_llm_engine.py / test_async_scheduling.py / basic_correctness / PP TEST_MODELS
# list hit (cleaned name=Llama-3.2-1B-Instruct path v3.2; the Aone /ppusw side has v3.1, so the
# red-zone main list v3.2 wins, consistent with the basic-correctness sh)
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
# test_async_scheduling.py (Qwen3-0.6B) + the light spec_decode eagle cases
# list hit: checkpoints/LLM/qwen/v3/Qwen3-0.6B
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
# test_context_length.py (JackFram/llama-160m)
# list hit: checkpoints/LLM/llama/v1.0/llama-160m
    "JackFram/llama-160m": "/nas_aisw/datasets/checkpoints/LLM/llama/v1.0/llama-160m",
# main model of test_struct_output_generate.py (xgrammar group)
# Ministral: Aone aliases hit LLM/Ministral/v1.0
    "mistralai/Ministral-8B-Instruct-2410":
        "/nas_aisw/datasets/checkpoints/LLM/Ministral/v1.0/Ministral-8B-Instruct-2410",
# Qwen2.5-1.5B: list hit checkpoints/LLM/qwen/v2.5/Qwen2.5-1.5B-Instruct
    "Qwen/Qwen2.5-1.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-1.5B-Instruct",
# used by struct_output.py (the file also has a speculative group, already excluded by -k)
    "meta-llama/Meta-Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Meta/v1.0/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-1.7B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B",
# deepseek distilled models inside struct_output.py
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/R1/DeepSeek-R1-Distill-Qwen-1.5B",

# ---- single: spec_decode eagle/mtp (the cases picked by -k; all hit Aone aliases or the list) ----
# Llama-3.1-8B base (main list path v3.1)
    "meta-llama/Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.1/Llama-3.1-8B-Instruct",
# eagle draft (yuhuili EAGLE/EAGLE3)
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3.1-Instruct-8B",
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE3/v3/EAGLE3-LLaMA3.1-Instruct-8B",
# qwen3 eagle3 group (the eagle3 cases)
    "Qwen/Qwen3-8B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-8B",
    "AngelSlim/Qwen3-8B_eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-8B_eagle3",
# deepseek_eagle group (eagle618 random small models)
    "eagle618/deepseek-v3-random":
        "/nas_aisw/datasets/checkpoints/LLM/deepseek/v3/deepseek-v3-random",
    "eagle618/eagle-deepseek-v3-random":
        "/nas_aisw/datasets/checkpoints/LLM/eagle/v3/eagle-deepseek-v3-random",
# mtp group
    "XiaomiMiMo/MiMo-7B-Base":
        "/nas_aisw/datasets/checkpoints/LLM/MiMo/v1.0/MiMo-7B-Base",
    "ZixiQi/DeepSeek-V3-4layers-MTP-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V3/DeepSeek-V3-4layers-MTP-FP8",
# Qwen3 variants used by the light eagle parametrization group
    "Qwen/Qwen3-1.7B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-1.7B-FP8",
    "Qwen/Qwen3-0.6B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B-FP8",
    "amd/PARD-Qwen3-0.6B":
        "/nas_aisw/datasets/checkpoints/LLM/PARD/v1.0/PARD-Qwen3-0.6B",
# speculator group (excluded by -k not *_speculator, link created in case of other references)
    "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-3.1-8B-Instruct-speculator.eagle3",
    "RedHatAI/Qwen3-8B-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3-8B-speculator.eagle3",
# spec_decode dataset (likaixin/InstructCoder, datasets path)
    "likaixin/InstructCoder":
        "/nas_aisw/datasets/datasets/LLM/likaixin/v1.0/InstructCoder",
# Llama-4-Scout eagle group (mostly _heavy/large and excluded by -k/mark, link as a fallback)
    "meta-llama/Llama-4-Scout-17B-16E-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/llama/v4/Llama-4-Scout-17B-16E-Instruct",
    "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-Llama-4-Scout-17B-16E-Instruct",
# VL eagle3 group (the matching cases are pytest.mark.skip; Rayzl hits, taobao-mnn pending staging)
    "Qwen/Qwen3-VL-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-7B-Instruct",
    "Rayzl/qwen2.5-vl-7b-eagle3-sgl":
        "/nas_aisw/datasets/checkpoints/LLM/qwen2.5/v1.0/qwen2.5-vl-7b-eagle3-sgl",
# taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3: MISS in both lists -> pending staging
# (the matching qwen3_vl_eagle3 cases are pytest.mark.skip, so not blocking; add the path once staged)
# "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3": "<NAS path once staged>",

    # ---- multi: distributed / DP / eagle_dp ----
# used by basic_correctness distributed as tiny-random (not in the list, Aone aliases hit tiny/v1.0)
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
    "hmellor/tiny-random-Gemma2ForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-Gemma2ForCausalLM",
# test_async_llm_dp.py (DP_SIZE=2) uses PowerMoE-3b
# HF org=ibm-research (referenced by the test); Aone aliases hit PowerMoE/v1.0
    "ibm-research/PowerMoE-3b":
        "/nas_aisw/datasets/checkpoints/LLM/PowerMoE/v1.0/PowerMoE-3b",
# test_eagle_dp.py reuses Llama-3.1-8B + EAGLE-LLaMA3.1-Instruct-8B above

# ---- multi: pipeline_parallel TEST_MODELS (picked by -k: no ray/Jamba/PowerLM/DeepSeek) ----
    "microsoft/Phi-3.5-MoE-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Phi/v1.0/Phi-3.5-MoE-instruct",
    "hmellor/Ilama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Ilama/v3.2/Ilama-3.2-1B",
    "intfloat/e5-mistral-7b-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/e5/v1/e5-mistral-7b-instruct",
    "BAAI/bge-multilingual-gemma2":
        "/nas_aisw/datasets/checkpoints/LLM/bge/v1.0/bge-multilingual-gemma2",
    "OpenGVLab/InternVL2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/InternVL/v1.0/InternVL2-1B",
    "microsoft/Phi-3.5-vision-instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Phi-3/v3.5/Phi-3.5-vision-instruct",
    "fixie-ai/ultravox-v0_5-llama-3_2-1b":
        "/nas_aisw/datasets/checkpoints/LLM/ultravox/v0/ultravox-v0_5-llama-3_2-1b",
# Note: ibm/PowerLM-3b, deepseek-ai/DeepSeek-V2-Lite-Chat and ai21labs/Jamba-tiny-dev among the
# PP TEST_MODELS are excluded by -k not(PowerLM/DeepSeek/Jamba), so no link is created.
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

root = ET.Element("testsuites", name="vLLM PPU Model Runner V2 (GHA)")
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

lines = ["### Model Runner V2 Test (PPU)", "",
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

# Mirrors the single/multi sections of aone_ci/scripts/test_area_ppu_model_runner_v2.sh:
# each step is its own pytest process; the per-step env (ENFORCE_EAGER / TARGET_TEST_SUITE /
# TP_SIZE / DP_SIZE / NCCL_CUMEM_HOST_ENABLE) is injected as a command prefix and only
# affects that _run_step call. Every step has shards=1 (no sharding, the tests use multi-card internally).
_run_single_steps() {
  _run_step "mrv2_core_llm_engine" 1 "${MRV2_CORE_LLM_ENGINE_ARGS[@]}"
  ENFORCE_EAGER=1 \
    _run_step "mrv2_core_async_scheduling" 1 "${MRV2_CORE_ASYNC_SCHEDULING_ARGS[@]}"
  _run_step "mrv2_core_context_length" 1 "${MRV2_CORE_CONTEXT_LENGTH_ARGS[@]}"
  _run_step "mrv2_core_min_tokens" 1 "${MRV2_CORE_MIN_TOKENS_ARGS[@]}"
  _run_step "mrv2_core_struct_output" 1 "${MRV2_CORE_STRUCT_OUTPUT_ARGS[@]}"
  _run_step "mrv2_spec_decode_e2e" 1 "${MRV2_SPEC_DECODE_E2E_ARGS[@]}"
}

_run_multi_steps() {
  TARGET_TEST_SUITE=L4 \
    _run_step "mrv2_distributed_basic_correctness" 1 \
      "${MRV2_DISTRIBUTED_BASIC_CORRECTNESS_ARGS[@]}"
  TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0 \
    _run_step "mrv2_distributed_dp" 1 "${MRV2_DISTRIBUTED_DP_ARGS[@]}"
  TP_SIZE=1 DP_SIZE=2 NCCL_CUMEM_HOST_ENABLE=0 \
    _run_step "mrv2_distributed_eagle_dp" 1 "${MRV2_DISTRIBUTED_EAGLE_DP_ARGS[@]}"
  _run_step "mrv2_pipeline_parallelism" 1 "${MRV2_PIPELINE_PARALLELISM_ARGS[@]}"
}

if [ "${MODE}" = "single" ]; then
  _run_single_steps
elif [ "${MODE}" = "multi" ]; then
  _run_multi_steps
else  # all
  _run_single_steps
  _run_multi_steps
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
