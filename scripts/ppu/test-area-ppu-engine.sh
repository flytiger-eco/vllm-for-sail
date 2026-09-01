#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-engine.sh — PPU Engine 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-engine.yml（容器内，cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/engine.yaml 的
# 迁移快照（见下方 ENGINE_SINGLE_*，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 无 multi 段（上游 V1 e2e 2/4 GPU
#                                      step 跑 spec_decode，PPU 单卡不适用）
#
# 机制移植自 aone_ci/scripts/test_area_ppu_engine.sh（该文件 AUTO-GENERATED
# 不可手改，故在此复刻）：
#   - single: 单进程 × 3 step（engine_basic / v1_engine / v1_e2e_general），
#     均限制可见 1 卡（对齐 Aone 1-PPU pod 语义：multi_gpu_test 用例在 1 卡下
#     自动 skip——test_engine_core_client.py num_gpus=4、test_engine_core.py
#     num_gpus=2 均靠此机制跳过）
#   - junit:  每 step 落 xml，EXIT trap 合并到 test-results/test.xml，
#     pytest 崩溃时也要补 error case（不能让 CI 信号失真）
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
TMP_JUNIT="/tmp/ppu-engine-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/engine.yaml single 段）
# ------------------------------------------------------------------------------
# Step 1: engine_basic — tests/engine/test_arg_utils.py + 根目录
#   test_sequence/logger/vllm_port（均为无 GPU 用例）。
#   快照时只取 test_arg_utils.py 而非整个 tests/engine/ 目录，原因（原 yaml 注释）：
#   - test_short_mm_context.py → llava-hf/llava-1.5-7b-hf 多模态，暂不覆盖
#   - test_config.py → 大量 ModelConfig() 调用下载 HF config，红区依赖重
ENGINE_SINGLE_BASIC_ARGS=(
  tests/engine/test_arg_utils.py
  tests/test_sequence.py
  tests/test_logger.py
  tests/test_vllm_port.py
)

# Step 2: v1_engine — tests/v1/engine/ 整目录（V1 engine core/client/async/
#   preprocess）。模块级 AutoTokenizer.from_pretrained（test_engine_core_client.py:55
#   Llama-3.2-1B-Instruct、test_engine_core.py:37 tiny-random-LlamaForCausalLM、
#   conftest.py:33 Llama-3.2-1B）由下方 MODEL_MAP symlink 解决，无需 ignore。
#
#   首跑实测（2026-08-27）：17 failed / 100 passed / 2 skipped（2732s）。
#   17 个 red 用例已逐个 deselect 标注（见下）；修复后删除对应行恢复。
ENGINE_SINGLE_V1_ARGS=(
  tests/v1/engine/
  # ---- deselect：首跑 red 用例（2026-08-27）----
  # abort 语义 ×6（final step abort / multi abort / abort final output）
  --deselect "tests/v1/engine/test_abort_final_step.py::test_abort_during_final_step[False]"
  --deselect "tests/v1/engine/test_abort_final_step.py::test_abort_during_final_step[True]"
  --deselect "tests/v1/engine/test_async_llm.py::test_multi_abort[RequestOutputKind.DELTA]"
  --deselect "tests/v1/engine/test_async_llm.py::test_multi_abort[RequestOutputKind.FINAL_ONLY]"
  --deselect "tests/v1/engine/test_async_llm.py::test_abort_final_output[RequestOutputKind.DELTA]"
  --deselect "tests/v1/engine/test_async_llm.py::test_abort_final_output[RequestOutputKind.FINAL_ONLY]"
  # EngineCore 基础 ×4
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_advanced_sampling"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_concurrent_batches"
  --deselect "tests/v1/engine/test_engine_core.py::test_engine_core_invalid_request_id_type"
  # encoder 零 kv-cache 实例 ×6
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_producer-0.01-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_consumer-0.7-True]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[False-ec_consumer-0.7-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_producer-0.01-False]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_consumer-0.7-True]"
  --deselect "tests/v1/engine/test_engine_core.py::test_encoder_instance_zero_kv_cache[True-ec_consumer-0.7-False]"
  # preprocess 错误处理 ×1
  --deselect "tests/v1/engine/test_preprocess_error_handling.py::test_preprocess_error_handling"
)

# Step 3: v1_e2e_general — tests/v1/e2e/general/（async scheduling、min_tokens、
#   streaming、context_length、sliding_window、cascade_attention）
ENGINE_SINGLE_E2E_ARGS=(
  tests/v1/e2e/general/
  # gemma-3n-E2B-it 模型较大，首版不覆盖（原 yaml 注释）
  --ignore=tests/v1/e2e/general/test_kv_sharing_fast_prefill.py
  # Qwen3-Next-80B-A3B-Instruct-FP8，PPU 单卡跑不了 80B（原 yaml 注释）
  --ignore=tests/v1/e2e/general/test_mamba_prefix_cache.py
  # Qwen3-Embedding-0.6B embedding model，首版 defer（原 yaml 注释）
  --ignore=tests/v1/e2e/general/test_pooling_chunked_prefill.py
  # JackFram/llama-160m 未入库红区（GHA 清单与 Aone aliases 均 MISS）；
  # 该文件仅 test_decoder_max_context_length_validation 一个参数化用例，
  # 整文件 ignore；模型入库后删除此行并在 MODEL_MAP 补路径
  --ignore=tests/v1/e2e/general/test_context_length.py
)

# multi：无 — 上游 V1 e2e 2/4 GPU steps 跑 spec_decode，PPU 单卡不适用

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 注意：禁止 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（上游
# TP 场景的 CUDA VMM workaround）——PPU 兼容层疑不支持 VMM API，是虚假
# OOM 头号嫌疑（详见 test-area-ppu-basic-correctness.sh 同段注释；
# Aone 侧从不设它且全绿）

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
# （NAS 绝对路径 = /nas_aisw/datasets/ + path）；清单未收录的取 Aone 侧
# ppu_model_aliases.json 同构路径（/ppusw/ → /nas_aisw/）。路径不存在时
# WARN 并跳过（该模型的用例会失败，日志里可见原因）。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # v1_engine conftest.py / test_engine_core_client.py 等
    # 清单命中：checkpoints_cleaned.json path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    # v1_engine conftest.py:33
    # 清单命中：path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B
    "meta-llama/Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B",
    # v1_engine / v1_e2e_general 多个用例
    # 清单命中：path=checkpoints/LLM/misc/v1.0/opt-125m
    "facebook/opt-125m": "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
    # test_engine_core.py:37 模块级 tokenizer
    # 清单未收录，与 basic-correctness 同源（runner 已 ls 确认存在）
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
    # v1_e2e_general（cascade_attention 等）
    # 清单 MISS，取 Aone aliases 同构路径
    "deepseek-ai/DeepSeek-V2-Lite":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite",
    # v1_e2e_general
    # 清单命中：path=checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct
    "Qwen/Qwen2-VL-2B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.0/Qwen2-VL-2B-Instruct",
    # v1_e2e_general
    # 清单命中：path=checkpoints/LLM/qwen/v3/Qwen3-0.6B
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
    # v1_e2e_general；清单 MISS，取 Aone aliases 同构路径（注意大写 Qwen 目录）
    "Qwen/Qwen2-1.5B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v1.0/Qwen2-1.5B-Instruct",
    # v1_e2e_general（min_tokens 等）；清单 MISS，取 Aone aliases 同构路径
    "bigcode/starcoder2-3b":
        "/nas_aisw/datasets/checkpoints/LLM/starcoder/v1.0/starcoder2-3b",
    # test_fast_incdec_prefix_err.py + test_correctness_sliding_window.py
    # 清单 MISS，取 Aone aliases 同构路径
    "google/gemma-3-1b-it":
        "/nas_aisw/datasets/checkpoints/LLM/gemma/v1.0/gemma-3-1b-it",
    # test_async_scheduling.py speculator；清单 MISS，取 Aone aliases 同构路径
    "nm-testing/Llama3_2_1B_speculator.eagle3":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/Llama3_2_1B_speculator.eagle3",
    # JackFram/llama-160m（test_context_length.py）：两个清单均 MISS，
    # 待入库；对应用例已在 ENGINE_SINGLE_E2E_ARGS 整文件 ignore（见上）：
    # "JackFram/llama-160m": "<NAS path 待入库后补>",
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

root = ET.Element("testsuites", name="vLLM PPU Engine (GHA)")
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

# ---- step summary：分 shard 统计表（markdown）。合并 test.xml 的
# testsuite name 已被改写为 label（丢失 shard 维度），故此处从原始
# shard xml 提取。宿主 workflow 把本文件 cat 进 GITHUB_STEP_SUMMARY，
# 在 run 的 Summary 页直接渲染（容器内拿不到该 env，需 workflow 接力）
SUMMARY = os.path.join(os.path.dirname(OUT), "summary.md")
COLS = ("tests", "failures", "errors", "skipped", "time")

def _stats(path):
    # junit 根节点 pytest 新旧版可能为 <testsuites> 或 <testsuite>
    root_ = ET.parse(path).getroot()
    suites = [root_] if root_.tag == "testsuite" else list(root_.iter("testsuite"))
    agg = dict.fromkeys(COLS, 0.0)
    for s in suites:
        for k in COLS:
            agg[k] += float(s.get(k) or 0)
    return agg

lines = ["### Engine Test (PPU)", "",
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
      # 注：不可写成 `[ $rc -ne 0 ] && rc_total=1` —— 条件为假时整个
      # 表达式返回 1，若它是函数/分支的最后一条命令，函数返回码变成 1，
      # 顶层 set -e 会在函数调用处杀掉脚本（测试全过反而 exit 1 的元凶）
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

if [ "${MODE}" = "single" ]; then
  # Aone single 是 1-PPU pod：限 1 卡使 multi_gpu_test 自动 skip，语义对齐
  CUDA_VISIBLE_DEVICES=0 _run_step "engine_basic" 1 "${ENGINE_SINGLE_BASIC_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_engine" 1 "${ENGINE_SINGLE_V1_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_e2e_general" 1 "${ENGINE_SINGLE_E2E_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area engine has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "engine_basic" 1 "${ENGINE_SINGLE_BASIC_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_engine" 1 "${ENGINE_SINGLE_V1_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "v1_e2e_general" 1 "${ENGINE_SINGLE_E2E_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
