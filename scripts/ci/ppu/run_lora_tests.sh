#!/bin/bash
# ==============================================================================
# scripts/ci/ppu/run_lora_tests.sh — PPU LoRA 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/pr-test-ppu.yml（容器内，cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/lora.yaml 的
# 迁移快照（见下方 LORA_SINGLE_ARGS / LORA_MULTI_ARGS，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single | multi   — 对应上游两个 lora 测试 job
#
# 机制移植自 aone_ci/scripts/test_area_ppu_lora.sh（该文件 AUTO-GENERATED 不可
# 手改，故在此复刻）：
#   - single: 4-shard 并发（CUDA_VISIBLE_DEVICES=i × pytest-shard）
#   - junit: 每 step/shard 落 xml，EXIT trap 合并到 test-results/test.xml，
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
TMP_JUNIT="/tmp/ppu-lora-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/lora.yaml single/multi 段）
# ------------------------------------------------------------------------------
# single = 上游 "LoRA %N" job（parallelism=4）的 PPU 版：
#   - 去掉 7 个 TP 文件（multi 侧跑）+ 3 个 PPU 已知不可跑文件
#   - OAM-810E 设备 ignore（快照自 device_conditional_ignores）：
#     fp8 kernel 需 SM≥8.9；triton fused_moe 精度问题。新设备接入时按
#     lora.yaml 的 per-device 语义增删下面两行。
LORA_SINGLE_ARGS=(
  tests/lora
  --ignore=tests/lora/test_chatglm3_tp.py
  --ignore=tests/lora/test_llama_tp.py
  --ignore=tests/lora/test_qwen3_with_multi_loras.py
  --ignore=tests/lora/test_olmoe_tp.py
  --ignore=tests/lora/test_deepseekv2_tp.py
  --ignore=tests/lora/test_gptoss_tp.py
  --ignore=tests/lora/test_qwen3moe_tp.py
  --ignore=tests/lora/test_minicpmv_tp.py
  # ChatGLM3 tokenizer._pad() 缺 padding_side 参数，与 transformers 不兼容
  --ignore=tests/lora/test_add_lora.py
  # whisper-small base model — whisper 在 PPU 上支持待验证
  --ignore=tests/lora/test_whisper.py
  # 模块级 snapshot_download，离线环境 collection 即崩（landmine）
  --ignore=tests/lora/test_default_mm_loras.py
  # PPU 推理精度差异，断言 pattern 不匹配，待 PPU 团队排查
  -k "not test_qwen2vl_multiple_lora_types"
  # --- OAM-810E device ignores ---
  # SM 8.0，fp8e4m3fn kernel 需 SM ≥ 8.9
  --ignore=tests/lora/test_punica_ops_fp8.py
  # triton fused_moe kernel 精度: Tensor mismatch 100%
  --ignore=tests/lora/test_fused_moe_lora_kernel.py
)

# multi = 上游 "LoRA TP (Distributed)" job 的 PPU 版（当前仅 1 个文件，
# TP=2；其余 TP 文件等模型 stage 到 /nas_aisw 后再放开）
LORA_MULTI_ARGS=(
  tests/lora/test_qwen3_with_multi_loras.py
)

# ------------------------------------------------------------------------------
# [env] 离线 + 运行时配置
# ------------------------------------------------------------------------------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM="false"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
# 上游 .buildkite/test_areas/lora.yaml TP 段的 OOM workaround，PPU 上无害
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
# tests/lora 里的模型名是硬编码 HF id；离线模式下 from_pretrained 走
# HF_HUB_CACHE 的标准目录结构（models--org--repo/snapshots/main）。NAS 上
# 预置的检查点在 /nas_aisw/datasets/checkpoints/LLM/ 下按家族分层，这里为
# 测试所需模型建 symlink（HF 标准布局，不依赖任何 fork 特有机制）。
# 路径不存在时 WARN 并跳过（该模型的用例会失败，日志里可见原因）。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # base model：single/multi 主力（conftest、test_lora_functions、
    # test_qwen3_unembed、test_worker、test_llm_with_multi_loras）
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
    # TODO: 按 NAS 实际结构补充 lora adapter，例如：
    # "charent/self_cognition_Alice":
    #     "/nas_aisw/datasets/checkpoints/LLM/lora/charent/self_cognition_Alice",
    # "charent/self_cognition_Bob":
    #     "/nas_aisw/datasets/checkpoints/LLM/lora/charent/self_cognition_Bob",
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

root = ET.Element("testsuites", name="vLLM PPU LoRA (GHA)")
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
  _run_step "lora_full" 4 "${LORA_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  _run_step "test_llm_with_multi_loras" 1 "${LORA_MULTI_ARGS[@]}"
else  # all
  _run_step "lora_full" 4 "${LORA_SINGLE_ARGS[@]}"
  _run_step "test_llm_with_multi_loras" 1 "${LORA_MULTI_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
