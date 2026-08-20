#!/bin/bash
# ==============================================================================
# scripts/ci/ppu/run_basic_correctness_tests.sh — PPU Basic Correctness 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-basic-correctness.yml（容器内，cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/basic_correctness.yaml 的
# 迁移快照（见下方 BC_SINGLE_* / BC_MULTI_ARGS，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single | multi   — 对应 Aone 两个 ptg-ai-test job
#
# 机制移植自 aone_ci/scripts/test_area_ppu_basic_correctness.sh（该文件
# AUTO-GENERATED 不可手改，故在此复刻）：
#   - single: 单进程 × 2 step，限制可见 1 卡（对齐 Aone 1-PPU pod 语义：
#     multi_gpu_test 用例在 1 卡下自动 skip，避免与 multi 段重复跑）
#   - multi:  单进程 -m distributed（测试内部 TP=2 自行用 2 卡）
#   - junit:  每 step 落 xml，EXIT trap 合并到 test-results/test.xml，
#     pytest 崩溃时也要补 error case（不能让 CI 信号失真）
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-bc-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有依赖：ray（multi 段 distributed_executor_backend="ray" 用例
# 需要）。镜像预装则跳过——不碰镜像已有栈；缺失才从 flytiger PyPI 补。
# ------------------------------------------------------------------------------
if python3 -c "import ray" 2>/dev/null; then
  echo "[deps] ray already installed: $(python3 -c 'import ray; print(ray.__version__)')"
else
  echo "[deps] installing ray from flytiger PyPI"
  python3 -m pip install --no-cache-dir ray \
    -i "https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"
fi

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/basic_correctness.yaml single/multi 段）
# ------------------------------------------------------------------------------
# single = Aone "basic-correctness single" job（1-PPU pod）：
#   - test_mem.py 全量（上游 3bb46975b "[XPU] transparent sleep mode support"
#     由 test_cumem.py 重命名而来，R085 并改为平台无关写法；旧基线分支仍叫
#     test_cumem.py，回退基线时此处改回）
#   - test_basic_correctness.py 全量（12 个 multi_gpu_test 在 1 卡下自动 skip；
#     本脚本 single 段统一 CUDA_VISIBLE_DEVICES=0 保持该语义）
BC_SINGLE_MEM_ARGS=(
  tests/basic_correctness/test_mem.py
  # test_deep_sleep_fp8_kvcache 显式 skip 并标注：其所需 Qwen/Qwen2-0.5B
  # 未入库红区（scripts/model_list 清单与 Aone aliases 均无）；且
  # requires_fp8 在 OAM-810E（SM 8.0，需 SM≥8.9）上本就 skip——-k 排除使
  # 该 skip 成为显式行为，不依赖平台能力检测；用例名以当前分支 test_mem.py 为准
  -k "not test_deep_sleep_fp8_kvcache"
)
BC_SINGLE_BASIC_ARGS=(
  tests/basic_correctness/test_basic_correctness.py
)
# test_cpu_offload.py 禁用（快照自原 yaml 注释段）：
#   OAM-810E 上 hmellor/tiny-random-LlamaForCausalLM（head_dim=4）触发 inductor
#   flex_attention NotImplementedError(head_dim<16)；vLLM 无全局 enforce-eager
#   env var，compare_two_settings arg1=[] 写死无法绕过。
#   恢复条件：fork conftest patch compare_two_settings 默认加 --enforce-eager，
#   或 PPU 团队 fix flex_attention head_dim<16。

# multi = Aone "basic-correctness multi" job（2-PPU pod）：distributed 子集
# （multi_gpu_test 真跑，TP=2；TARGET_TEST_SUITE 默认 L4，A100 组自动 skip）
BC_MULTI_ARGS=(
  tests/basic_correctness/test_basic_correctness.py
  -m
  distributed
)

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 注意：禁止 export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（上游
# TP 场景的 CUDA VMM workaround）——PPU 兼容层疑不支持 VMM API，是虚假
# OOM 头号嫌疑：96 GiB free 时 20 MiB 分配失败且 free>total 统计错乱
# （本 area single uni EngineCore 与 lora multi TP rank1 两案例均发生在
# 设此 env 的 GHA 环境；Aone 侧从不设它且全绿，DEC-0013 当时明确决定
# 不引入）。删除重跑验证；若虚假 OOM 仍现再查 PPU SDK/驱动

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
# 路径优先取红区已存模型清单 scripts/model_list/*.json 的 path 字段
# （NAS 绝对路径 = /nas_aisw/datasets/ + path）；清单未收录的按 Aone
# /ppusw 同构路径标注"待确认"。路径不存在时 WARN 并跳过（该模型的用例
# 会失败，日志里可见原因）。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # test_mem.py::test_end_to_end（pytorch checkpoint 组）+
    # test_basic_correctness.py::test_failed_model_execution / distributed 组
    # 清单命中：checkpoints_cleaned.json path=checkpoints/LLM/misc/v1.0/opt-125m
    "facebook/opt-125m": "/nas_aisw/datasets/checkpoints/LLM/misc/v1.0/opt-125m",
    # test_basic_correctness.py MODELS + distributed 组
    # 清单命中：path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct
    # （清单 ms_name 为 unsloth/...，按 name 匹配；Aone /ppusw 侧是 v3.1 目录，
    # 以红区清单的 v3.2 为准）
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    # test_mem.py（safetensors 组 / deep_sleep 组）+ test_vllm_gc_ed +
    # test_basic_correctness.py distributed 组
    # 清单未收录，已在 runner 上 ls 确认存在（tiny/v1.0/ 共 4 个 tiny 模型）
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
    # test_basic_correctness.py MODELS；同上，已 ls 确认
    "hmellor/tiny-random-Gemma2ForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-Gemma2ForCausalLM",
    # Qwen/Qwen2-0.5B：仅 test_mem.py::test_deep_sleep_fp8_kvcache 使用，
    # 未入库红区 → 该用例已在 BC_SINGLE_MEM_ARGS 用 -k 显式 skip（见上）；
    # 未来入库后移除该 -k 并在此补路径：
    # "Qwen/Qwen2-0.5B": "<NAS path 待入库后补>",
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

root = ET.Element("testsuites", name="vLLM PPU Basic Correctness (GHA)")
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
PYEOF
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
      CUDA_VISIBLE_DEVICES="${shard}" pytest -v -s "${args[@]}" \
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
    pytest -v -s "${args[@]}" --junit-xml="${out_xml}"
    local rc=$?
    set -e
    echo "[step] ${label} rc=${rc}"
    if [ "${rc}" -ne 0 ]; then TOTAL_RC=1; fi
  fi
}

if [ "${MODE}" = "single" ]; then
  # Aone single 是 1-PPU pod：限 1 卡使 multi_gpu_test 自动 skip，语义对齐
  CUDA_VISIBLE_DEVICES=0 _run_step "test_mem" 1 "${BC_SINGLE_MEM_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "test_basic_correctness" 1 "${BC_SINGLE_BASIC_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  _run_step "test_basic_correctness_distributed" 1 "${BC_MULTI_ARGS[@]}"
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "test_mem" 1 "${BC_SINGLE_MEM_ARGS[@]}"
  CUDA_VISIBLE_DEVICES=0 _run_step "test_basic_correctness" 1 "${BC_SINGLE_BASIC_ARGS[@]}"
  _run_step "test_basic_correctness_distributed" 1 "${BC_MULTI_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
