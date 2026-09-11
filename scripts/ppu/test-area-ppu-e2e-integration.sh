#!/bin/bash
# [ci-smoke] 第二批 12 area PR 门禁全量验证触碰行（本 PR 勿合并）
# ==============================================================================
# scripts/ppu/test-area-ppu-e2e-integration.sh — PPU E2E Integration 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-e2e-integration.yml（K8s worker Pod 内，
#         cwd = /workspace/source）。
#
# 用例基于 tests/evals/gsm8k/test_gsm8k_correctness.py 的 GSM8K 精度框架：
#   config yaml 定义 model + server_args + accuracy_threshold，conftest.py 经
#   --config-list-file 参数化（config 文件在 tests/evals/gsm8k/configs/）。
#
# 环境变量：
#   TEST_MODE          all(默认) | single | multi
#   PPU_DEVICE_LABEL   设备标签（默认 OAM-810E）；用于复刻 device_conditional_ignores
#
# 注意（GSM8K 离线数据）：tests/evals/gsm8k/gsm8k_eval.py 运行时从
#   raw.githubusercontent.com 下载 train/test.jsonl 到 /tmp。红区 Pod 不通外网，
#   下方 [setup] pre-cache 段从 /nas_aisw 候选路径预置 /tmp/{train,test}.jsonl
#   （与 test-area-ppu-lm-eval.sh 一致）；NAS 未入库时 WARN 且 GSM8K 用例会
#   因 ConnectionError 失败（恢复条件：jsonl 入库到任一候选路径）。
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
# [setup] pre-cache GSM8K jsonl（红区 pod 不通 raw.githubusercontent.com）
# ------------------------------------------------------------------------------
# gsm8k_eval.py:download_and_cache_file() 用 requests.get 直接拉 GitHub raw
# 的 {train,test}.jsonl 到 /tmp/{train,test}.jsonl（os.path.exists 命中即复用）。
# 红区无外网 → ConnectionError → 所有 GSM8K 用例 fail。HF_HUB_OFFLINE 只 hook
# huggingface_hub，不影响 requests，无法绕开此下载。与 lm-eval 脚本同款逻辑：
# 从 /nas_aisw 预置卷候选路径探测拷贝，缺失时 WARN（用例将 fail，日志可见原因）。
# Run 34594819557 补齐：此前本脚本漏掉此段（lm-eval 有），e2e 两 step 必挂。
echo "========== [setup] pre-cache gsm8k jsonl =========="
for f in train.jsonl test.jsonl; do
  dst="/tmp/${f}"
  if [ -f "${dst}" ]; then
    echo "[gsm8k] ${dst} already present"
    continue
  fi
  for _cand in \
    "/nas_aisw/datasets/gsm8k/${f}" \
    "/nas_aisw/datasets/eval/gsm8k/${f}" \
    "/nas_aisw/datasets/grade-school-math/${f}"; do
    if [ -f "${_cand}" ]; then
      cp "${_cand}" "${dst}" && echo "[gsm8k] cached ${f} <- ${_cand}"
      break
    fi
  done
  if [ ! -f "${dst}" ]; then
    echo "[gsm8k] WARN: ${f} not found on /nas_aisw candidates — GSM8K tests will fail (待入库)"
  fi
done

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/e2e_integration.yaml single/multi 段）
# ------------------------------------------------------------------------------
# Run 34594819557 修复：models-ppu-e2e-{single,multi,fp8}.txt 及引用的 3 个
# yaml（DeepSeek-V2-Lite-Prefetch-Offload / DeepSeek-V2-Lite-EP-EPLB /
# Qwen3-30B-A3B-FP8-EP-EPLB）已补入 tests/evals/gsm8k/configs/（复刻 Aone
# 侧已验证版本）。此前 7 个文件缺失，conftest 无参数化源 → 收集 1 个裸
# test_gsm8k_correctness 报 fixture 'config_filename' not found（2 step 全挂）。
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
