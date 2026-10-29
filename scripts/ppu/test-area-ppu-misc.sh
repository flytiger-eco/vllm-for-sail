#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-misc.sh -- PPU Miscellaneous tests (GitHub Actions)
# ------------------------------------------------------------------------------
# Caller: .github/workflows/ppu-nightly-ci.yml (in the container, cwd = /workspace).
#
# Environment variables:
#   TEST_MODE   all(default) | single   - this area has no multi section (the misc PPU
#                                      scope has no multi_gpu_test, single PPU only)
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
TMP_JUNIT="/tmp/ppu-misc-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# No area-specific pip dependencies: aone_ci/scripts/test_area_ppu_misc.sh only installs the
# pytest toolchain (pytest-asyncio/tblib/pytest-shard), already installed by
# ppu_install_dependency.sh. The upstream misc steps that need kv_connectors/opentelemetry/
# lm-eval are all skipped or ignored in the PPU scope (see [tests] below), so nothing to add.

# ------------------------------------------------------------------------------
# [tests] case selection (snapshot of the single section in aone_ci/ppu_extras/misc.yaml, 5 labels)
# ------------------------------------------------------------------------------
# misc PPU scope = the GPU subset of upstream .buildkite/test_areas/misc.yaml:
#   ✅ Step 1 V1 Others (GPU) → v1/{core,executor,kv_offload,sample,
#      logits_processors,worker,spec_decode,kv_connector/unit,metrics} +
#      v1/test_{oracle,request,outputs}.py (split into 4 steps to isolate interference)
#   ✅ Step 7 Async Engine, Inputs, Utils → detokenizer/multimodal/utils_
#   X the remaining steps (CPU-only / Regression / Examples / 2GPU tracing /
#      python-only / H100 batch-invariance / acceptance-length) are outside the PPU scope.
# Known failures are excluded precisely with --ignore= plus a note on the reproducing run and
# root cause (matching the original yaml).

# Step 1a: V1 Core (GPU)
MISC_V1_CORE_GPU_ARGS=(
  tests/v1/core
  -m
"not cpu_test"  # excludes the 9 CPU-only files
  # Run 50086632: EngineDeadError on PPU (Landmine #20)
  --ignore=tests/v1/core/test_scheduler_e2e.py
)

# Step 1b: V1 Inference Components
MISC_V1_INFERENCE_ARGS=(
  tests/v1/executor
  tests/v1/kv_offload
  tests/v1/sample
  tests/v1/logits_processors
  tests/v1/worker
  # Run 51113338: EngineDeadError on PPU (Landmine #20)
  --ignore=tests/v1/sample/test_sampling_params_e2e.py
  # Run 51113338: Server exited unexpectedly on PPU (Landmine #20)
  --ignore=tests/v1/logits_processors/test_custom_online.py
  # Run 51113338: lm_eval version incompatible (enforce_eager /
  # local-completions not registered in CI image lm_eval)
  --ignore=tests/v1/sample/test_logprobs_e2e.py
  # Run 51408355: spec decode logprob precision mismatch on PPU
# (diff ~0.22, tol=0.1) - PPU floating point drift
  --ignore=tests/v1/sample/test_logprobs.py
)

# ---- Run 34594819557 failure clusters excluded (v1_inference, 74 cases, grouped by file) ----
# (a) the kv_offload/cpu pair, 44 cases: OSError [Errno 22] cannot create the shm shared
#     memory region (same root cause as the already ignored test_offloading_connector)
#     plus swap_blocks_batch cuMemcpyBatchAsync error 998 (the PPU batch copy path is
#     unsupported). test_gpu_worker.py keeps 4 passing cases and
#     test_shared_offload_region.py keeps test_wait_for_file_size_* 3 cases PASSED (they
#     do not touch the shm/batch copy path), hence the precise deselect instead of a file
#     ignore. Restore: once the PPU pod shm setup and the batch copy kernel are adapted.
# (b) the logits_processors pair, 13 cases: test_correctness 6 cases hit Cannot re-initialize
#     CUDA in forked subprocess (a forked child re-initialises the PPU device);
#     test_custom_offline 7 cases hit EngineCore startup failure / fork CUDA pollution
#     (landmine #20 class of engine startup crash; the FQCN-pooling one only exposed the
#     same root cause in run 34602096292 and is added here).
# (c) worker/test_gpu_model_runner.py, 17 cases: 12 ERROR = the granite-4.0-tiny-preview NAS
#     snapshot lacks model.safetensors (HF offline resolve fails); 5 FAILED = Only dense CPU
#     tensors can be pinned (the PPU pin_memory limitation).
# (d) worker/test_gpu_worker_weight_transfer.py, 1 case: sparse weight transfer mixes the
#     cuda:0 and cpu devices.
MISC_V1_INFERENCE_DESELECTS=(
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-1-1024-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-1-1024-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-1-512-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-1-512-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-3-1024-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[False-cuda:0-0-4-256-64-3-512-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-1-1024-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-1-1024-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-1-512-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-1-512-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-3-1024-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-3-1024-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-3-512-3-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer[True-cuda:0-0-4-256-64-3-512-3-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-1-1024-2-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-1-1024-2-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-1-512-2-False]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-1-512-2-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-3-1024-2-True]"
  "tests/v1/kv_offload/cpu/test_gpu_worker.py::test_transfer_multi_group[cuda:0-0-256-64-3-512-2-True]"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_cleanup_after_create_next_view_releases_mmap"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_cleanup_creator_all_effects"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_cleanup_idempotent"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_cleanup_non_creator_all_effects"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_cumulative_overflow_raises"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_cursor_advances"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_exact_fill_succeeds"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_multi_tensor_layout"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_multiprocess_slots"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_overflow_does_not_mutate_cursor"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_row_stride_with_multiple_workers"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_shape_and_stride"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_single_overflow_raises"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_storage_offset_rank0"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_storage_offset_rank1"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_worker_isolation"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_create_next_view_write_visible_in_raw_mmap"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_creator_flag_set_on_first_open"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_file_exists_after_construction"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_file_has_correct_size"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_joiner_flag_not_set"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_multi_worker_race_exactly_one_creator"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_multi_worker_race_shared_memory_visible"
  "tests/v1/kv_offload/cpu/test_shared_offload_region.py::test_multiprocess_race_construct_and_write"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test0-50-cuda:0]"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test1-50-cuda:0]"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test2-50-cuda:0]"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test3-50-cuda:0]"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test4-50-cuda:0]"
  "tests/v1/logits_processors/test_correctness.py::test_logitsprocs[logitsprocs_under_test5-50-cuda:0]"
  "tests/v1/logits_processors/test_custom_offline.py::test_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_CLASS-pooling]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_CLASS-spec_dec]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT-pooling]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_ENTRYPOINT-spec_dec]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_FQCN-pooling]"
  "tests/v1/logits_processors/test_custom_offline.py::test_rejects_custom_logitsprocs[CustomLogitprocSource.LOGITPROC_SOURCE_FQCN-spec_dec]"
  "tests/v1/worker/test_gpu_model_runner.py::test_get_nans_in_logits"
  "tests/v1/worker/test_gpu_model_runner.py::test_hybrid_attention_mamba_tensor_shapes"
  "tests/v1/worker/test_gpu_model_runner.py::test_hybrid_cache_integration"
  "tests/v1/worker/test_gpu_model_runner.py::test_init_kv_cache_with_kv_sharing_valid"
  "tests/v1/worker/test_gpu_model_runner.py::test_init_kv_cache_without_kv_sharing"
  "tests/v1/worker/test_gpu_model_runner.py::test_kv_cache_stride_order"
  "tests/v1/worker/test_gpu_model_runner.py::test_load_model_weights_inplace"
  "tests/v1/worker/test_gpu_model_runner.py::test_mamba_cache_raises_when_max_num_seqs_exceeds_blocks"
  "tests/v1/worker/test_gpu_model_runner.py::test_reload_weights_before_load_model"
# Run 38547348048: an earlier case leaks the default torch CUDA device, so a CPU pinned
# tensor is built on CUDA and triggers Only dense CPU tensors can be pinned.
  "tests/v1/worker/test_gpu_model_runner.py::test_reasoning_config_without_custom_logitsprocs_does_not_need_output_token_ids"
  "tests/v1/worker/test_kv_block_zeroer.py::test_block_ids_are_not_overwritten_while_copy_is_in_flight"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_config"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_new_request"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_no_changes"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_pp_async_multi_request_keeps_rank_state_consistent"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_pp_non_async_multi_request_keeps_token_buffers_consistent"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_request_finished"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_request_resumed"
  "tests/v1/worker/test_gpu_model_runner.py::test_update_states_request_unscheduled"
  "tests/v1/worker/test_gpu_worker_weight_transfer.py::test_update_weights_sparse_dispatches_to_sparse_receive"
)
for _t in "${MISC_V1_INFERENCE_DESELECTS[@]}"; do
  MISC_V1_INFERENCE_ARGS+=(--deselect "${_t}")
done
unset _t

# Step 1c: V1 Speculative Decoding
MISC_V1_SPEC_DECODE_ARGS=(
  tests/v1/spec_decode
  -m
"not slow_test"  # excludes the large-model acceptance length cases
  # Run 49439357: EAGLE spec decode engine crash on PPU (Landmine #20)
  --ignore=tests/v1/spec_decode/test_max_len.py
  # Run 49769845: model not cached on PPU CI machine
  # - test_speculators_eagle3: RedHatAI/Qwen3-8B-quantized.w4a16
  #   and RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 not staged
# (upstream removed test_tree_attention.py, so the dead ignore is gone too)
  --ignore=tests/v1/spec_decode/test_speculators_eagle3.py
  # Run 50086632: eagle3 acceptance length AttributeError on PPU
  --ignore=tests/v1/spec_decode/test_acceptance_length.py
# ---- Run 34594819557 failure clusters excluded (v1_spec_decode) ----
# dflash configs raise a pydantic ValidationError (incompatible pydantic version in the pod):
# test_dflash_lookahead.py 3/3 fail -> whole file ignored
  --ignore=tests/v1/spec_decode/test_dflash_lookahead.py
# test_mtp.py 2/2 fail: PosixPath/NoneType TypeError (MTP path resolution)
  --ignore=tests/v1/spec_decode/test_mtp.py
# test_speculators_correctness.py 2/2 fail: the dflash/peagle speculator
# models are not staged (HF offline LocalEntryNotFoundError)
  --ignore=tests/v1/spec_decode/test_speculators_correctness.py
# test_eagle.py has 52 cases and only this one fails: same dflash pydantic incompatibility
  --deselect tests/v1/spec_decode/test_eagle.py::test_set_inputs_first_pass_dflash
)

# Step 1d: V1 KV Connector + Metrics + Standalone
MISC_V1_CONNECTORS_METRICS_ARGS=(
  tests/v1/kv_connector/unit
  tests/v1/metrics
  tests/v1/test_oracle.py
  tests/v1/test_request.py
  tests/v1/test_outputs.py
  -m
  "not cpu_test"
  # Landmine #20: engine core init crash on PPU
  --ignore=tests/v1/metrics/test_engine_logger_apis.py
  --ignore=tests/v1/metrics/test_ray_metrics.py
  # Run 51408355: Ray 2.43.0 required, CI image has 2.31.0
  --ignore=tests/v1/kv_connector/unit/test_nixl_connector.py
  # Run 51408355: NIXL library not available in CI image
  --ignore=tests/v1/kv_connector/unit/test_nixl_connector_hma.py
# Run 34580922394: creating the CPU offload shared memory region fails with OSError [Errno 22]
# (vllm/v1/kv_offload/cpu/shared_offload_region.py:100); all 8 cases of the file fail
# (test_cpu_offloading x6 + test_tiering/fs_tiering_offloading).
# Restore: unignore once the PPU pod /dev/shm or the shared_offload_region adaptation is fixed.
  --ignore=tests/v1/kv_connector/unit/test_offloading_connector.py
# Run 34580922394: the scheduler/worker event sequences of the two ExampleConnector cases
# disagree (PPU execution ordering), single cases excluded.
  --deselect tests/v1/kv_connector/unit/test_multi_connector.py::test_multi_example_connector_consistency
# Run 38547348048: the test stub does not initialise _has_mamba, a new attribute read by production code.
  --deselect tests/v1/kv_connector/unit/test_nixl_push_connector.py::TestPushWriterMlaReplication::test_mla_hetero_tp_writes_every_d_rank
)

# Step 7: Async Engine, Inputs, Utils, Worker (GPU)
MISC_ASYNC_ENGINE_UTILS_ARGS=(
  tests/detokenizer
  tests/multimodal
  tests/utils_
  -m
"not cpu_test"  # excludes the CPU-only multimodal tests
  # Run 49439357: model meta-llama/llama-2-7b-hf path mismatch
  --ignore=tests/detokenizer/test_stop_strings.py
  # Landmine #17: module-level import of missing packages (librosa)
  --ignore=tests/multimodal/media/test_audio.py
  # Run 49217147: PlaceholderModule runtime librosa import
  --ignore=tests/multimodal/test_audio.py
  # Run 49217147: red zone cannot reach external URLs (bogotobogo.com)
  --ignore=tests/multimodal/media/test_connector.py
  # No module named 'vllm_test_utils'
  --ignore=tests/utils_/test_mem_utils.py
  # No module named 'timm'
  --ignore=tests/utils_/test_tensor_schema.py
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
# HF ids referenced from tests/{v1,detokenizer,multimodal,utils_} (grepped with double quotes)
# checked one by one against the red-zone list: prefer the list's path field (marked cleaned),
# then the /ppusw->/nas_aisw mapping of aone_ci/scripts/ppu_model_aliases.json (marked aone).
# All 38 ids hit, none pending staging.
# Large models (8B/20B and up) are mostly used by the already --ignore'd spec_decode/slow/cpu
# cases; creating the symlink is harmless; when missing, [setup] logs MISS and skips.
    "amd/PARD-Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/PARD/v1.0/PARD-Llama-3.2-1B",  # aone
    "BAAI/bge-base-en-v1.5":
        "/nas_aisw/datasets/checkpoints/LLM/BAAI/v1.5/bge-base-en-v1.5",  # cleaned
    "distilbert/distilgpt2":
        "/nas_aisw/datasets/checkpoints/LLM/gpt/v1.0/distilgpt2",  # cleaned
    "facebook/opt-125m":
        "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",  # cleaned
    "facebook/opt-350m":
        "/nas_aisw/datasets/checkpoints/LLM/opt/v1.0/opt-350m",  # aone
    "google/gemma-3-1b-it":
        "/nas_aisw/datasets/checkpoints/LLM/gemma/v1.0/gemma-3-1b-it",  # aone
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",  # aone
    "ibm-granite/granite-4.0-tiny-preview":
        "/nas_aisw/datasets/checkpoints/LLM/granite/v1.0/granite-4.0-tiny-preview",  # aone
    "llava-hf/llava-1.5-7b-hf":
        "/nas_aisw/datasets/checkpoints/LLM/llava-hf/v1.5/llava-1.5-7b-hf",  # cleaned
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf":
        "/nas_aisw/datasets/checkpoints/LLM/llava/v1.0/llava-onevision-qwen2-0.5b-ov-hf",  # aone
    "llava-hf/llava-v1.6-mistral-7b-hf":
        "/nas_aisw/datasets/checkpoints/LLM/llava/v1.6/llava-v1.6-mistral-7b-hf",  # aone
    "meta-llama/llama-2-7b-hf":
        "/nas_aisw/datasets/checkpoints/LLM/llama/v2.0/llama-2-7b-hf",  # cleaned
    "meta-llama/Llama-3.1-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.1/Llama-3.1-8B-Instruct",  # cleaned
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",  # cleaned
    "meta-llama/Meta-Llama-3-8B":
        "/nas_aisw/datasets/checkpoints/LLM/Meta/v1.0/Meta-Llama-3-8B",  # cleaned
    "meta-llama/Meta-Llama-3-8B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/llama/v3/Meta-Llama-3-8B-Instruct",  # cleaned
    "nm-testing/Llama3_2_1B_speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/Llama3_2_1B_speculator.eagle3",  # aone
    "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v3.0/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",  # aone
    "nm-testing/Speculator-Qwen3-30B-MOE-VL-Eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v3.0/Speculator-Qwen3-30B-MOE-VL-Eagle3",  # aone
    "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v3.0/Speculator-Qwen3-8B-Eagle3-converted-071-quantized",  # aone
    "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized-w4a16":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v3.0/Speculator-Qwen3-8B-Eagle3-converted-071-quantized-w4a16",  # aone
    "openai/gpt-oss-20b":
        "/nas_aisw/datasets/checkpoints/LLM/gpt/v1/gpt-oss-20b",  # cleaned
    "Qwen/Qwen1.5-7B":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen1.5/v1.0/Qwen1.5-7B",  # aone
    "Qwen/Qwen2-0.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen2-0.5B-Instruct",  # cleaned
    "Qwen/Qwen2.5-1.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-1.5B-Instruct",  # cleaned
    "Qwen/Qwen2.5-VL-3B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-VL-3B-Instruct",  # cleaned
    "Qwen/Qwen2-VL-2B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct",  # cleaned
    "Qwen/Qwen3-0.6B":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",  # cleaned
    "Qwen/Qwen3-8B":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-8B",  # cleaned
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3.0/Qwen3-VL-30B-A3B-Instruct-FP8",  # aone
    "RedHatAI/gpt-oss-20b-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/gpt/v1.0/gpt-oss-20b-speculator.eagle3",  # aone
    "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v1.0/Llama-3.1-8B-Instruct-speculator.eagle3",  # aone
# The NAS directory name ends with a dot: Qwen. (copied from the list path, not a typo)
    "RedHatAI/Qwen2.5-VL-3B-Instruct-quantized.w8a8":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen./v1.0/Qwen2.5-VL-3B-Instruct-quantized.w8a8",  # aone
    "RedHatAI/Qwen3-8B-speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen3-8B-speculator.eagle3",  # aone
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/TinyLlama-1.1B-Chat-v1.0",  # cleaned
    "XiaomiMiMo/MiMo-7B-Base":
        "/nas_aisw/datasets/checkpoints/LLM/MiMo/v1.0/MiMo-7B-Base",  # aone
    "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE3/v3/EAGLE3-LLaMA3.1-Instruct-8B",  # aone
    "yuhuili/EAGLE-LLaMA3.1-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3.1-Instruct-8B",  # aone
    "yuhuili/EAGLE-LLaMA3-Instruct-8B":
        "/nas_aisw/datasets/checkpoints/LLM/EAGLE/v1.0/EAGLE-LLaMA3-Instruct-8B",  # aone
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

root = ET.Element("testsuites", name="vLLM PPU Misc (GHA)")
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

lines = ["### Misc Test (PPU)", "",
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

# misc is single-only (5 independent steps, each limited to 1 card); there is no multi section.
_run_single_steps() {
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_core_gpu" 1 "${MISC_V1_CORE_GPU_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_inference" 1 "${MISC_V1_INFERENCE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_spec_decode" 1 "${MISC_V1_SPEC_DECODE_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_connectors_metrics" 1 "${MISC_V1_CONNECTORS_METRICS_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "async_engine_utils" 1 "${MISC_ASYNC_ENGINE_UTILS_ARGS[@]}"
}

if [ "${MODE}" = "single" ]; then
  _run_single_steps
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area misc has no multi-mode steps configured" >&2
  exit 2
else  # all
  _run_single_steps
fi

# ------------------------------------------------------------------------------
# [summary] aggregate exit code: the script exits with the merged rc so CI reports the truth
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
