#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-entrypoints.sh — PPU Entrypoints 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-entrypoints.yml（容器内，
# cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/entrypoints.yaml
# 的迁移快照并按 v0.23 上游目录布局修正（见下方 EP_SINGLE_UNIT_ARGS，调整
# 用例直接改这里）。模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw +
# HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 无 multi 段（上游 entrypoints
#                                      无 multi-GPU step）
#
# 机制移植自 aone_ci/scripts/test_area_ppu_entrypoints.sh（AUTO-GENERATED 不可
# 手改，故在此复刻）：单进程单 step，junit EXIT trap 合并 + 崩溃补 error case。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|multi|all) ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}'" >&2; exit 2 ;;
esac

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-entrypoints-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/entrypoints.yaml single 段）
# ------------------------------------------------------------------------------
# Step 1: entrypoints_unit — 上游 "Entrypoints Unit Tests" 的 PPU 版：
#   tool_parsers + tests/entrypoints/ 非 LLM 子集（无 model 依赖可快速验证）。
#   scope 决策（原 yaml 注释）：API Server 1/2（重度 model 依赖 + server
#   startup）、Pooling、Responses API、OpenAI API Correctness 首版 defer；
#   LLM 相关归独立 area entrypoints_llm，此处不重复覆盖。
#
#   ignore 集对齐上游 .buildkite/test_areas/entrypoints.yaml unit step：
#   上游忽略 llm/openai/serve/test_chat_utils.py/pooling/speech_to_text/
#   generate；PPU 另忽略 weight_transfer。原快照的 rpc/instrumentator/
#   offline_mode/sagemaker 四条 ignore 在 v0.23 布局下均为死路径
#   （instrumentator、sagemaker 已迁至 serve/ 子目录，被 serve ignore 覆盖；
#   rpc、offline_mode 目录不存在），已删除。
EP_SINGLE_UNIT_ARGS=(
  tests/entrypoints/openai/tool_parsers
  tests/entrypoints/
  --ignore=tests/entrypoints/llm
  --ignore=tests/entrypoints/openai
  --ignore=tests/entrypoints/test_chat_utils.py
  --ignore=tests/entrypoints/pooling
  # serve — 上游归独立 "API Server 2" step（server startup 重度 model 依赖，
  # 对齐 API Server defer 决策）；v0.23 起 instrumentator/sagemaker 等
  # 均迁入该子树
  --ignore=tests/entrypoints/serve
  # speech_to_text — 上游归独立 "Speech to Text" step；whisper 在 PPU 支持
  # 未验证（同 test-area-ppu-lora.sh 对 test_whisper.py 的处理），且
  # correctness 文件有模块级 get_tokenizer(whisper-large-v3)（离线 collection
  # 即炸）+ 需 HF 数据集/evaluate wer metric，离线成本过高，首版 defer
  --ignore=tests/entrypoints/speech_to_text
  # generate — 上游归 "API Server openai Part 2" step（对齐 defer 决策）
  --ignore=tests/entrypoints/generate
  # weight_transfer — LLM(NCCL weight transfer)，PPU 上未验证
  --ignore=tests/entrypoints/weight_transfer
)

# 注：原快照 Step 2（v1_entrypoints，跑 tests/v1/entrypoints/）已删除——
# v0.23 上游移除该目录（用例迁入 tests/entrypoints/openai/，归 defer 的
# API Server scope；上游 entrypoints.yaml 亦无 "Entrypoints V1" step），
# 保留必挂（pytest exit 4: file or directory not found）。

# multi：无 — 上游 entrypoints 无 multi-GPU step

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
#
# 注：原快照为 v1/entrypoints 准备的 symlink（opt-125m、llava-1.5-7b、
# DeepSeek-R1-Distill、Meta-Llama-3.1-8B、Ministral、Qwen2.5-1.5B、
# Qwen3-1.7B、Qwen2.5-VL-3B、tiny-random-Llama）已随 Step 2 一并移除——
# 其消费用例均在已 ignore 的 openai/serve 子树内，当前 scope 无消费者；
# 后续放开 API Server scope 时按 git 历史加回即可。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # ---- tool_parsers（tokenizer 级依赖，不起引擎） ----
    # test_hermes_tool_parser.py 等；清单命中 Llama/v3.2
    "meta-llama/Llama-3.2-1B-Instruct":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B-Instruct",
    # test_hermes_tool_parser.py::test_qwen3 系；清单命中 qwen/v3
    "Qwen/Qwen3-32B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-32B",
    # test_granite4_tool_parser.py；清单 MISS，取 Aone aliases 同构路径
    "ibm-granite/granite-4.0-h-tiny":
        "/nas_aisw/datasets/checkpoints/LLM/granite/v1.0/granite-4.0-h-tiny",
    # test_hermes_tool_parser.py LoRA tokenizer；清单 MISS，Aone 同构路径
    "minpeter/LoRA-Llama-3.2-1B-tool-vllm-ci":
        "/nas_aisw/datasets/checkpoints/LLM/LoRA/v1.0/LoRA-Llama-3.2-1B-tool-vllm-ci",
    # ---- anthropic（引擎级，RemoteOpenAIServer） ----
    # anthropic/test_messages.py；清单命中 qwen/v3
    "Qwen/Qwen3-0.6B": "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-0.6B",
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

root = ET.Element("testsuites", name="vLLM PPU Entrypoints (GHA)")
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

lines = ["### Entrypoints Test (PPU)", "",
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

if [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area entrypoints has no multi-mode steps configured" >&2
  exit 2
else
  # single / all：Step 1 纯 CPU 单测为主（tool_parsers + 轻量 entrypoints），
  # anthropic 用例起 1 个 Qwen3-0.6B server；统一限 1 卡对齐 Aone 1-PPU
  # pod 语义
  CUDA_VISIBLE_DEVICES=0 _run_step "entrypoints_unit" 1 "${EP_SINGLE_UNIT_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
