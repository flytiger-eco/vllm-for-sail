#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-e2e-integration.sh — PPU E2E Integration 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-e2e-integration.yml（K8s worker Pod 内，
#         cwd = /workspace/source）。
#
# 完全自包含，运行时不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/e2e_integration.yaml
# 的迁移快照（见下方 E2E_SINGLE_ARGS / E2E_MULTI_ARGS / E2E_FP8_ARGS，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（ppu-distributed-action 默认挂载 + HF_HUB_CACHE）。
#
# 用例基于 tests/evals/gsm8k/test_gsm8k_correctness.py 的 GSM8K 精度框架：
#   config yaml 定义 model + server_args + accuracy_threshold，conftest.py 经
#   --config-list-file 参数化（config 文件在 tests/evals/gsm8k/configs/）。
#
# 环境变量：
#   TEST_MODE          all(默认) | single | multi  — 对应 Aone 两个 ptg-ai-test job
#   PPU_DEVICE_LABEL   设备标签（默认 OAM-810E）；用于复刻 device_conditional_ignores
#
# 机制移植自 aone_ci/scripts/test_area_ppu_e2e_integration.sh（该文件
# AUTO-GENERATED 不可手改，故在此复刻）：
#   - single: 1 step e2e_prefetch_offload（DeepSeek-V2-Lite 1 GPU，限 1 卡）
#   - multi:  2 step 顺序执行——e2e_ep_eplb（DeepSeek-V2-Lite TP=2 DP=2，用满 4 卡）
#             + e2e_fp8_ep_eplb（Qwen3-30B-A3B-FP8；OAM-810E 不支持 FP8，--ignore 跳过）
#   - junit:  每 step 落 xml，EXIT trap 合并到 test-results/test.xml，
#     pytest 崩溃时也要补 error case（不能让 CI 信号失真）
#
# 注意（GSM8K 离线数据）：tests/evals/gsm8k/gsm8k_eval.py 运行时从
#   raw.githubusercontent.com 下载 train/test.jsonl 到 /tmp。红区 Pod 不通外网，
#   需预置 /tmp/{train,test}.jsonl（数据仅 vendor 在 aone_ci/data/gsm8k/，为遵守
#   "运行时零依赖 aone_ci/" 本脚本不引用该路径；数据 vendoring 到 fork 可用位置
#   作为独立后续项处理，否则依赖 GSM8K 的用例会因 ConnectionError 失败）。
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

# 设备标签（workflow 经 extra_env 注入 PPU_DEVICE_LABEL=OAM-810E）；复刻 aone sh
# 的第 2 参数语义，用于 device_conditional_ignores（OAM-810E 跳过 FP8）。
DEVICE_LABEL="${PPU_DEVICE_LABEL:-OAM-810E}"
echo "[device] label: ${DEVICE_LABEL}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-e2e-integration-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有依赖：无。
# ------------------------------------------------------------------------------
# GSM8K 框架依赖（aiohttp/numpy/regex/requests/tqdm/pyyaml）由镜像预装；pytest
# 工具链（pytest-asyncio/tblib/pytest-shard/pyyaml）已在 ppu_install_dependency.sh
# 统一安装（对位 aone sh 的 pip install pytest-asyncio tblib pytest-shard），
# 故此处无需再补装。

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/e2e_integration.yaml single/multi 段）
# ------------------------------------------------------------------------------
# single = Aone "e2e-integration single" job（1-PPU pod）：
#   - e2e_prefetch_offload：DeepSeek-V2-Lite prefetch offload 精度验证（1 GPU）;
#     GSM8K eval 框架，threshold=0.25，200 questions（本脚本 single 段
#     CUDA_VISIBLE_DEVICES=0 限 1 卡，对齐 Aone 1-PPU pod 语义）
E2E_SINGLE_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-single.txt
)

# multi = Aone "e2e-integration multi" job（4-PPU pod），2 step 顺序执行：
#   - e2e_ep_eplb：DeepSeek-V2-Lite EP+EPLB 精度验证（4 GPU, TP=2 DP=2）;
#     GSM8K eval 框架，threshold=0.25，200 questions
E2E_MULTI_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-multi.txt
)
#   - e2e_fp8_ep_eplb：Qwen3-30B-A3B-FP8 EP+EPLB 精度验证（4 GPU, FP8）;
#     OAM-810E 不支持 FP8 → 快照自原 yaml device_conditional_ignores，
#     DEVICE_LABEL=OAM-810E 时追加 --ignore 整文件跳过（下方 mode dispatch 处理）。
E2E_FP8_ARGS=(
  tests/evals/gsm8k/test_gsm8k_correctness.py
  --config-list-file=configs/models-ppu-e2e-fp8.txt
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
# 路径优先取红区已存模型清单 scripts/ppu/model_alises/*.json 的 path 字段
# （NAS 绝对路径 = /nas_aisw/datasets/ + path）；清单未收录的按 Aone
# /ppusw 同构路径标注"待确认"。路径不存在时 WARN 并跳过（该模型的用例
# 会失败，日志里可见原因）。
echo "========== [setup] HF cache symlinks (/nas_aisw models) =========="
python3 - <<'PYEOF'
import os

MODEL_MAP = {
    # e2e_prefetch_offload（single）+ e2e_ep_eplb（multi）：DeepSeek-V2-Lite。
    # 清单 scripts/ppu/model_alises/*.json 未收录，Aone ppu_model_aliases.json
    # 命中 /ppusw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite
    # （/ppusw → /nas_aisw）。
    "deepseek-ai/DeepSeek-V2-Lite":
        "/nas_aisw/datasets/checkpoints/LLM/DeepSeek/V2/DeepSeek-V2-Lite",
    # e2e_fp8_ep_eplb（multi）：Qwen3-30B-A3B-FP8。OAM-810E 不支持 FP8，该 step
    # 在本设备上 --ignore 整文件跳过，模型实际不会加载；路径仍登记备用。
    # 清单命中：checkpoints_cleaned.json path=checkpoints/LLM/qwen/v3/Qwen3-30B-A3B-FP8
    "Qwen/Qwen3-30B-A3B-FP8":
        "/nas_aisw/datasets/checkpoints/LLM/qwen/v3/Qwen3-30B-A3B-FP8",
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

root = ET.Element("testsuites", name="vLLM PPU E2E Integration (GHA)")
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

lines = ["### E2E Integration Test (PPU)", "",
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
  # worker Pod 与编排 runner 容器共享同一 NAS（/mnt/wl_nas ↔ /wl_nas），宿主
  # workflow 的 Publish test summary 步骤读取并 cat 进 GITHUB_STEP_SUMMARY。
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

# ------------------------------------------------------------------------------
# [run/skip] e2e_fp8_ep_eplb：复刻 aone sh 的 device_conditional_ignores。
# OAM-810E 不支持 FP8 → aone sh 用 --ignore 整文件跳过（0 用例收集）。此处直接
# 记为 skipped（写 skipped junit + 登记 label），既保留 aone 的“跳过”意图，
# 又避免 pytest “no tests collected”(exit 5) 把整轮误判为 FAIL；其它设备
# （非 OAM-810E）经 _run_step 正常执行该 step。
# ------------------------------------------------------------------------------
_run_fp8_step() {
  local label="e2e_fp8_ep_eplb"
  case "${DEVICE_LABEL}" in
    OAM-810E)
      echo "========== [step] ${label} (skipped: OAM-810E 不支持 FP8) =========="
      STEP_LABELS_LIST="${STEP_LABELS_LIST} ${label}"
      python3 - "${TMP_JUNIT}/${label}.xml" "${label}" <<'PYEOF'
import sys
from xml.etree import ElementTree as ET

out, label = sys.argv[1], sys.argv[2]
ts = ET.Element("testsuite", name=label, tests="1", errors="0",
                failures="0", skipped="1", time="0")
tc = ET.SubElement(ts, "testcase", name=label,
                   classname=f"gha_ci.{label}", time="0")
ET.SubElement(tc, "skipped",
              message="OAM-810E does not support FP8 "
                      "(device_conditional_ignores)")
ET.ElementTree(ts).write(out, encoding="UTF-8", xml_declaration=True)
print(f"[step] {label} skipped (device={out})")
PYEOF
      ;;
    *)
      _run_step "${label}" 1 "${E2E_FP8_ARGS[@]}"
      ;;
  esac
}

if [ "${MODE}" = "single" ]; then
  # Aone single 是 1-PPU pod：DeepSeek-V2-Lite prefetch offload 用 1 卡，
  # 限 CUDA_VISIBLE_DEVICES=0 对齐 Aone 1-PPU pod 语义
  CUDA_VISIBLE_DEVICES=0 _run_step "e2e_prefetch_offload" 1 "${E2E_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  # e2e_ep_eplb：DeepSeek-V2-Lite TP=2 DP=2 内部用满 4 卡（不限 CUDA_VISIBLE_DEVICES）
  _run_step "e2e_ep_eplb" 1 "${E2E_MULTI_ARGS[@]}"
  _run_fp8_step
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "e2e_prefetch_offload" 1 "${E2E_SINGLE_ARGS[@]}"
  _run_step "e2e_ep_eplb" 1 "${E2E_MULTI_ARGS[@]}"
  _run_fp8_step
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
