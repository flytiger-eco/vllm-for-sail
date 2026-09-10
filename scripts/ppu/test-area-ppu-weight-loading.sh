#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-weight-loading.sh — PPU Weight Loading 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-weight-loading.yml（容器内，cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/weight_loading.yaml 的
# 迁移快照（见下方 WL_* 配置数组，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single | multi   — 对应 Aone 两个 ptg-ai-test job
#
# 机制移植自 aone_ci/scripts/test_area_ppu_weight_loading.sh（该文件
# AUTO-GENERATED 不可手改，故在此复刻）：
#   - single: 无 step（上游 weight_loading 全部用例都是 TP=2，single 段为空）
#   - multi:  遍历 6 个 model config（quant/model/revision 经 env var 下传），
#     每个 config 单进程跑同一个 test_weight_loading.py，测试内部硬编码
#     tensor_parallel_size=2 自行用 2 卡
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
TMP_JUNIT="/tmp/ppu-weight-loading-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/weight_loading.yaml multi 段）
# ------------------------------------------------------------------------------
# 上游 weight_loading 用 run_model_weight_loading_test.sh 迭代 models.txt
# （27 个 model config），每个 config 用 env var(MODEL_NAME/QUANTIZATION/
# REVISION) 驱动同一个 test_weight_loading.py（内部硬编码 tensor_parallel_size=2）。
# PPU 侧复刻该机制：per-config env var + 单进程 pytest。
#
# 红区选集（6 个，全 TP=2，均有 NAS 别名可跑）：
#   wl_gptq_marlin_zephyr    gptq_marlin  robertgshaw2/zephyr-7b-beta-channelwise-gptq
#   wl_gptq_marlin_tinyllama gptq_marlin  TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
#   wl_gptq_tinyllama        gptq         TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
#   wl_ct_w8w8_static        compressed-tensors  nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change
#   wl_ct_w8a8_dynamic       compressed-tensors  nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2
#   wl_none_fairseq2         None         mgleize/fairseq2-dummy-Llama-3.2-1B
#
# 排除（快照自原 yaml 可用性分析，均因红区不满足条件而不跑）：
#   - fp8 模型（nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test、
#     neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV）：有别名但 PPU OAM-810E
#     SM8.0 < FP8 所需 SM8.9，跳过。
#   - 未 staged 无别名（TheBloke/Llama-2-7B-GPTQ、TechxGenus/gemma-1.1-2b-it-GPTQ、
#     nm-testing/tinyllama-oneshot-w4a16-group128-v2、
#     nm-testing/tinyllama-oneshot-w8a16-per-channel、
#     nm-testing/Phi-3-mini-128k-instruct-FP8、
#     neuralmagic/Phi-3-medium-128k-instruct-quantized.w4a16、
#     casperhansen/mixtral-instruct-awq 等）：红区无预置检查点，跳过。
#   - capability>=90 的 SparseLlama 2of4 组：既无别名又要求 SM9.0，跳过。
#   - revision 变体（gptq-8bit-32g 等）：model redirect 单路径只映射 main 权重，
#     变体需独立 staged 检查点，仅跑 main。
#
# 模块级 snapshot_download 审计：CLEAN —— test_weight_loading.py 从 env var 读
# 模型名，无模块级下载，vllm_runner 负责加载。
WL_LABELS=(
  wl_gptq_marlin_zephyr
  wl_gptq_marlin_tinyllama
  wl_gptq_tinyllama
  wl_ct_w8w8_static
  wl_ct_w8a8_dynamic
  wl_none_fairseq2
)
WL_QUANT=(
  gptq_marlin
  gptq_marlin
  gptq
  compressed-tensors
  compressed-tensors
  None
)
WL_MODEL=(
  robertgshaw2/zephyr-7b-beta-channelwise-gptq
  TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
  TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ
  nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change
  nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2
  mgleize/fairseq2-dummy-Llama-3.2-1B
)
WL_REV=(main main main main main main)
WL_ARGS=(tests/weight_loading/test_weight_loading.py)

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
    # 路径均取自 aone_ci/scripts/ppu_model_aliases.json（/ppusw/ → /nas_aisw/）；
    # 主清单 scripts/ppu/model_alises/*.json 未收录这几个 weight_loading 专用
    # 量化小模型，按 Aone 别名表拼 NAS 路径。
    # wl_gptq_marlin_zephyr
    "robertgshaw2/zephyr-7b-beta-channelwise-gptq":
        "/nas_aisw/datasets/checkpoints/LLM/zephyr/v1.0/"
        "zephyr-7b-beta-channelwise-gptq",
    # wl_gptq_marlin_tinyllama + wl_gptq_tinyllama
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/"
        "TinyLlama-1.1B-Chat-v1.0-GPTQ",
    # wl_ct_w8w8_static
    "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v1.0/"
        "tinyllama-oneshot-w8w8-test-static-shape-change",
    # wl_ct_w8a8_dynamic
    "nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2":
        "/nas_aisw/datasets/checkpoints/LLM/optimization/v2/"
        "tinyllama-oneshot-w8a8-dynamic-token-v2",
    # wl_none_fairseq2
    "mgleize/fairseq2-dummy-Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/fairseq2/v1.0/"
        "fairseq2-dummy-Llama-3.2-1B",
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

root = ET.Element("testsuites", name="vLLM PPU Weight Loading (GHA)")
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

lines = ["### Weight Loading Test (PPU)", "",
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

# weight_loading 全部用例都是 TP=2（multi 段）：single 无 step；multi/all 跑全部
# 6 个 model config。每个 config 通过命令前缀 env var 下传 MODEL_NAME/
# QUANTIZATION/REVISION（pytest 子进程继承），单进程执行（shards=1，不限
# CUDA_VISIBLE_DEVICES → 2 卡全可见，测试内部 tensor_parallel_size=2 自行用满）。
_run_weight_loading_configs() {
  local i
  for i in "${!WL_LABELS[@]}"; do
    MODEL_NAME="${WL_MODEL[$i]}" QUANTIZATION="${WL_QUANT[$i]}" \
      REVISION="${WL_REV[$i]}" \
      _run_step "${WL_LABELS[$i]}" 1 "${WL_ARGS[@]}"
  done
}

if [ "${MODE}" = "single" ]; then
  echo "[run] no single steps configured for this area (all cases are TP=2)"
elif [ "${MODE}" = "multi" ]; then
  _run_weight_loading_configs
else  # all
  _run_weight_loading_configs
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
