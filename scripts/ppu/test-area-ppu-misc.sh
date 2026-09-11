#!/bin/bash
# [ci-smoke] 第二批 12 area PR 门禁全量验证触碰行（本 PR 勿合并）
# ==============================================================================
# scripts/ppu/test-area-ppu-misc.sh — PPU Miscellaneous 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-misc.yml（容器内，cwd = /workspace）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 无 multi 段（misc PPU scope
#                                      无 multi_gpu_test，纯 single PPU）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-misc-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# 无 area 特有 pip 依赖：aone_ci/scripts/test_area_ppu_misc.sh 仅装 pytest
# 工具链（pytest-asyncio/tblib/pytest-shard），已由 ppu_install_dependency.sh
# 统一安装；上游 misc 的 kv_connectors/opentelemetry/lm-eval 等依赖对应的
# step 在 PPU scope 内均被跳过或 ignore（见下方 [tests] 段），故无需补装。

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/misc.yaml single 段，5 个 label）
# ------------------------------------------------------------------------------
# misc PPU scope = 上游 .buildkite/test_areas/misc.yaml 的 GPU 子集：
#   ✅ Step 1 V1 Others (GPU) → v1/{core,executor,kv_offload,sample,
#      logits_processors,worker,spec_decode,kv_connector/unit,metrics} +
#      v1/test_{oracle,request,outputs}.py（拆成 4 个 step 隔离干扰）
#   ✅ Step 7 Async Engine, Inputs, Utils → detokenizer/multimodal/utils_
#   ❌ 其余 step（CPU-only / Regression / Examples / 2GPU tracing /
#      python-only / H100 batch-invariance / acceptance-length）不在 PPU scope
# 已知失败一律用 --ignore= 精确排除并注释复现 run/根因（对齐原 yaml）。

# Step 1a: V1 Core (GPU)
MISC_V1_CORE_GPU_ARGS=(
  tests/v1/core
  -m
  "not cpu_test"  # 排除 9 个 CPU-only 文件
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
  # (diff ~0.22, tol=0.1) — PPU 浮点精度偏差
  --ignore=tests/v1/sample/test_logprobs.py
)

# ---- Run 34594819557 失败聚类排除（v1_inference，74 例，按文件分组）----
# (a) kv_offload/cpu 两文件 44 例：OSError [Errno 22] shm 共享内存区创建失败
#     （与已 ignore 的 test_offloading_connector 同根因）+ swap_blocks_batch
#     cuMemcpyBatchAsync error 998（PPU 批量拷贝路径未支持）。
#     test_gpu_worker.py 幸存 4 例、test_shared_offload_region.py 幸存
#     test_wait_for_file_size_* 3 例 PASSED（不触 shm/批量拷贝路径），
#     故精确 deselect 而非整文件 ignore。
#     恢复条件：PPU pod shm 配置与批量拷贝 kernel 适配修复。
# (b) logits_processors 两文件 12 例：test_correctness 6 例 Cannot
#     re-initialize CUDA in forked subprocess（fork 子进程重复初始化 PPU
#     设备）；test_custom_offline 6 例 EngineCore failed to start
#     （Landmine #20 同类引擎启动崩溃）。
# (c) worker/test_gpu_model_runner.py 17 例：12 ERROR = granite-4.0-tiny-
#     preview NAS 快照缺 model.safetensors（HF offline resolve 失败）；
#     5 FAILED = Only dense CPU tensors can be pinned（PPU pin_memory 限制）。
# (d) worker/test_gpu_worker_weight_transfer.py 1 例：稀疏权重传输
#     cuda:0 vs cpu 设备不一致。
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
  "not slow_test"  # 排除大模型 acceptance length
  # Run 49439357: EAGLE spec decode engine crash on PPU (Landmine #20)
  --ignore=tests/v1/spec_decode/test_max_len.py
  # Run 49769845: model not cached on PPU CI machine
  # - test_speculators_eagle3: RedHatAI/Qwen3-8B-quantized.w4a16
  #   and RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 not staged
  # （上游已删除 test_tree_attention.py，原死 ignore 一并移除）
  --ignore=tests/v1/spec_decode/test_speculators_eagle3.py
  # Run 50086632: eagle3 acceptance length AttributeError on PPU
  --ignore=tests/v1/spec_decode/test_acceptance_length.py
  # ---- Run 34594819557 失败聚类排除（v1_spec_decode）----
  # dflash 配置类 pydantic ValidationError（pod 内 pydantic 版本不兼容）：
  # test_dflash_lookahead.py 3/3 全挂 → 整文件 ignore
  --ignore=tests/v1/spec_decode/test_dflash_lookahead.py
  # test_mtp.py 2/2 全挂：PosixPath/NoneType TypeError（MTP 路径解析）
  --ignore=tests/v1/spec_decode/test_mtp.py
  # test_speculators_correctness.py 2/2 全挂：dflash/peagle speculator
  # 模型未 stage（HF offline LocalEntryNotFoundError）
  --ignore=tests/v1/spec_decode/test_speculators_correctness.py
  # test_eagle.py 52 例中仅此例挂：同属 dflash pydantic 不兼容
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
  # Run 34580922394: CPU offload 共享内存区创建失败 OSError [Errno 22]
  # （vllm/v1/kv_offload/cpu/shared_offload_region.py:100），该文件 8/8 用例全挂
  # （test_cpu_offloading×6 + test_tiering/fs_tiering_offloading）。
  # 恢复条件：PPU pod /dev/shm 或 shared_offload_region 适配修复后 unignore。
  --ignore=tests/v1/kv_connector/unit/test_offloading_connector.py
  # Run 34580922394: 两个 ExampleConnector 的 scheduler/worker 事件序列断言
  # 不一致（PPU 执行时序差异），单用例排除。
  --deselect tests/v1/kv_connector/unit/test_multi_connector.py::test_multi_example_connector_consistency
)

# Step 7: Async Engine, Inputs, Utils, Worker (GPU)
MISC_ASYNC_ENGINE_UTILS_ARGS=(
  tests/detokenizer
  tests/multimodal
  tests/utils_
  -m
  "not cpu_test"  # 排除 CPU-only multimodal 测试
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
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# 默认离线（模型走 /nas_aisw 预置卷）；需要在线下载时设 PPU_TEST_ONLINE=1
if [[ "${PPU_TEST_ONLINE:-0}" != "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

# HF 缓存：workflow 已注入 HF_HUB_CACHE=/nas_aisw/datasets/hf_cache/hub；
# 未注入时（本地调试）探测 /nas_aisw 下的候选路径
if [ -z "${HF_HUB_CACHE:-}" ]; then
  for _cand in /nas_aisw/datasets/hf_cache/hub "$HOME/.cache/huggingface/hub"; do
    if [ -d "${_cand}" ]; then
      export HF_HUB_CACHE="${_cand}"
      break
    fi
  done
fi
echo "[env] HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}"

# PPU SDK: Triton/Inductor 编译需要 cuda.h + ptxas + libcuda（缺失时
# torch.compile 类测试 BackendCompilerFailed）
PPU_SDK_DIR="/usr/local/PPU_SDK/CUDA_SDK"
if [ -d "${PPU_SDK_DIR}" ]; then
  export C_INCLUDE_PATH="${PPU_SDK_DIR}/include:${C_INCLUDE_PATH:-}"
  export LIBRARY_PATH="${PPU_SDK_DIR}/lib64:${LIBRARY_PATH:-}"
  export CUDA_PATH="${PPU_SDK_DIR}"
  export PATH="${PPU_SDK_DIR}/bin:${PATH}"
fi

# ------------------------------------------------------------------------------
# [setup] HF hub cache symlink：HF id → /nas_aisw 本地检查点
# ------------------------------------------------------------------------------
# 路径优先取红区已存模型清单 scripts/ppu/model_alises/*.json 的 path 字段
# （NAS 绝对路径 = /nas_aisw/datasets/ + path）；清单未收录的按 Aone
# /ppusw 同构路径标注"待确认"。路径不存在时 WARN 并跳过。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # 快照自 tests/{v1,detokenizer,multimodal,utils_} 引用的 HF id（双引号
    # grep）逐个对照红区清单：优先 scripts/ppu/model_alises/*.json 的 path
    # 字段（标 cleaned），其次 aone_ci/scripts/ppu_model_aliases.json
    # 的 /ppusw→/nas_aisw（标 aone）。38 个 id 全部命中，无「待入库」。
    # 大模型（8B/20B 等）多用于已 --ignore 的 spec_decode/slow/cpu 用例，
    # 建 symlink 无害；NAS 缺失时 [setup] 段打 MISS 并跳过。
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
    # NAS 目录名带尾点 'Qwen.'（照抄清单 path，非笔误）
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
# [junit] 合并基础设施（含 pytest 崩溃兜底）
# ------------------------------------------------------------------------------
STEP_LABELS_LIST=""

# shellcheck disable=SC2329  # 只由下方 `trap _emit_junit EXIT` 调用
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

# ---- step summary：分 shard 统计表（markdown）
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
  # 分片 step 的 pytest 输出重定向到 TMP_JUNIT，一并收进 artifact 便于排障
  cp -f "${TMP_JUNIT}"/*.log "${RESULTS_DIR}/" 2>/dev/null || true
  # 容器以 root 运行，产物须可被 runner 用户读取（upload-artifact）
  chmod -R a+rwX "${RESULTS_DIR}" 2>/dev/null || true
  # [k8s-summary] 经 NAS 回传 per-unit 统计表：RESULTS_DIR 在 Pod 内、回收即失；
  if [ -n "${PPU_SUMMARY_NAS_DIR:-}" ] && [ -f "${RESULTS_DIR}/summary.md" ]; then
    if mkdir -p "${PPU_SUMMARY_NAS_DIR}" 2>/dev/null && \
       cp -f "${RESULTS_DIR}/summary.md" "${PPU_SUMMARY_NAS_DIR}/summary.md"; then
      echo "[k8s-summary] summary.md -> ${PPU_SUMMARY_NAS_DIR}/summary.md"
    else
      echo "[k8s-summary] WARN: NAS 回传失败（${PPU_SUMMARY_NAS_DIR} 不可写？）"
    fi
  fi
}
trap _emit_junit EXIT

# ------------------------------------------------------------------------------
# [run] 单 step 执行器：shards>1 → 并发分片；shards==1 → 单进程
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

# misc 是 single-only（5 个独立 step，各限 1 卡跑）；无 multi 段。
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
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
