#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-samplers.sh — PPU Samplers 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-samplers.yml（容器内，cwd = /workspace）。
#
# 完全自包含，不依赖 aone_ci/。用例选集是 aone_ci/ppu_extras/samplers.yaml 的
# 迁移快照（见下方 SAMPLERS_SINGLE_ARGS，调整用例直接改这里）。
# 模型走 /nas_aisw 预置卷（docker -v /nas_aisw:/nas_aisw + HF_HUB_CACHE）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 无 multi 段（tests/samplers/
#                                      无 @multi_gpu_test，纯 single PPU）
#
# 机制移植自 aone_ci/scripts/test_area_ppu_samplers.sh（AUTO-GENERATED 不可
# 手改，故在此复刻）：单进程单 step，junit EXIT trap 合并 + 崩溃补 error case。
#
# 已知 follow-up（原 yaml 注释保留）：
#   - Flashinfer 变体：上游跑 2 次（第 2 次 VLLM_USE_FLASHINFER_SAMPLER=1），
#     Aone 侧 sh_renderer 无 per-step env 字段故未拆；GHA 侧如需可加第 2 个
#     step（env 在 _run_step 调用处 export 即可）。
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
TMP_JUNIT="/tmp/ppu-samplers-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/samplers.yaml single 段）
# ------------------------------------------------------------------------------
# samplers_default = 上游 "Samplers Test" step 的 PPU 版（4 file 跑 2 个）：
SAMPLERS_SINGLE_ARGS=(
  tests/samplers
  # test_beam_search.py：TinyLlama/TinyLlama-1.1B-Chat-v1.0 未 stage（原 yaml
  # 注释）。注意：该模型现已入红区清单（checkpoints/LLM/TinyLlama/v1.0/...），
  # MODEL_MAP 已建 symlink；beam search 在 PPU 的正确性尚未验证，首跑稳定后
  # 可删除本行放开
  --ignore=tests/samplers/test_beam_search.py
  # test_no_bad_words.py（2026-05-28，原 yaml 注释）：setup_method 直接调
  # transformers AutoTokenizer.from_pretrained(MODEL)，绕过 vLLM 的
  # maybe_model_redirect → Aone 红区 cache miss → 2/8 fail。
  # GHA 侧用 HF 标准缓存布局 symlink（见下方 MODEL_MAP），from_pretrained
  # 离线直读缓存、不经 vllm redirect，原缺口结构上不存在；distilgpt2 的 NAS
  # 路径未经清单证实（Aone aliases 同构），保守起见本版仍保持 ignore，
  # 首跑确认 symlink MISS/OK 后可删除本行验证放开
  --ignore=tests/samplers/test_no_bad_words.py
)

# multi：无（samplers area 无 multi_gpu_test，纯 single PPU）

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
    # test_logprobs.py + test_ignore_eos.py
    # 清单 MISS，取 Aone aliases 同构路径
    "distilbert/distilgpt2":
        "/nas_aisw/datasets/checkpoints/LLM/gpt/v1.0/distilgpt2",
    # test_ignore_eos.py（Aone 侧曾 redirect 到 Instruct；GHA symlink 机制
    # 直接映射本名，清单命中 path=checkpoints/LLM/Llama/v3.2/Llama-3.2-1B）
    "meta-llama/Llama-3.2-1B":
        "/nas_aisw/datasets/checkpoints/LLM/Llama/v3.2/Llama-3.2-1B",
    # test_beam_search.py（当前 ignore，模型已入清单，放开后即用）：
    # 清单命中 path=checkpoints/LLM/TinyLlama/v1.0/TinyLlama-1.1B-Chat-v1.0
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        "/nas_aisw/datasets/checkpoints/LLM/TinyLlama/v1.0/TinyLlama-1.1B-Chat-v1.0",
    # test_no_bad_words.py（当前 ignore，放开后即用）；与 basic-correctness
    # 同源（runner 已 ls 确认存在）
    "hmellor/tiny-random-LlamaForCausalLM":
        "/nas_aisw/datasets/checkpoints/LLM/tiny/v1.0/tiny-random-LlamaForCausalLM",
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

root = ET.Element("testsuites", name="vLLM PPU Samplers (GHA)")
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

lines = ["### Samplers Test (PPU)", "",
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

if [ "${MODE}" = "single" ]; then
  CUDA_VISIBLE_DEVICES=0 _run_step "samplers_default" 1 "${SAMPLERS_SINGLE_ARGS[@]}"
elif [ "${MODE}" = "multi" ]; then
  echo "[mode] ERROR: area samplers has no multi-mode steps configured" >&2
  exit 2
else  # all
  CUDA_VISIBLE_DEVICES=0 _run_step "samplers_default" 1 "${SAMPLERS_SINGLE_ARGS[@]}"
fi

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
