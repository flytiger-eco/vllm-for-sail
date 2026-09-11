#!/bin/bash
# ==============================================================================
# scripts/ppu/test-area-ppu-models-multimodal.sh — PPU Models Multimodal 测试执行（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/test-area-ppu-models-multimodal.yml（容器内，cwd = /workspace）。
#
# 环境变量：
#   TEST_MODE   all(默认) | single   — 本 area 仅 single（对位上游 Multi-Modal
#               Models (Standard) core_model 子集）；multimodal distributed 已在
#               models_distributed area 覆盖，故无 multi 段
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

MODE="${TEST_MODE:-all}"
case "${MODE}" in
  single|all) ;;
  # 本 area 无 multi 段（multimodal distributed 由 models_distributed area 覆盖）；
  # 显式拒绝而非静默通过，避免 CI 假绿。
  multi) echo "[mode] ERROR: area models-multimodal 无 multi 段（multimodal distributed 已在 models_distributed area 覆盖）" >&2; exit 2 ;;
  *) echo "[mode] ERROR: invalid TEST_MODE '${MODE}' (本 area 仅 single|all)" >&2; exit 2 ;;
esac

# workflow_dispatch 的 pytest_args 透传：按空白切分后追加到每个 step 的
# pytest 命令尾部（如 `-k test_foo -x`），排障时缩小范围而不必改脚本。
read -ra PYTEST_EXTRA <<< "${PYTEST_EXTRA_ARGS:-}"

RESULTS_DIR="${REPO_ROOT}/test-results"
TMP_JUNIT="/tmp/ppu-models-multimodal-junit"
mkdir -p "${RESULTS_DIR}" "${TMP_JUNIT}"

# ------------------------------------------------------------------------------
# [deps] area 特有依赖：opencv（cv2）。
# ------------------------------------------------------------------------------
if python3 -c "import cv2" 2>/dev/null; then
  echo "[deps] opencv already importable: $(python3 -c 'import cv2; print(cv2.__version__)')"
else
  echo "[deps] installing opencv-python-headless from flytiger PyPI"
  python3 -m pip install --no-cache-dir opencv-python-headless \
    -i "https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple" || \
    echo "[deps] WARN: opencv install failed — test_common import 链会在 collection 崩溃"
fi

# ------------------------------------------------------------------------------
# [tests] 用例选集（快照自 aone_ci/ppu_extras/models_multimodal.yaml single 段）
# ------------------------------------------------------------------------------
# Step mm_generation_dedicated —— 对位上游 Multi-Modal Models (Standard) 1-4 中
# 非 test_common 部分：tests/models/multimodal -m core_model，排除 test_common
# 及一批离线/依赖不满足的文件。非 ignore 的文件里实际带 core_model marker 的仅
# test_nemotron_parse.py（nvidia/NVIDIA-Nemotron-Parse-v1.1）与
# pooling/test_llava_next.py（royokong/e5-v，见 MODEL_MAP）。
# 每条 --ignore 的原因逐条保留自原 yaml：
MM_GENERATION_ARGS=(
  tests/models/multimodal
  -m
  core_model
  # collection-time crash：test_common.py module-level video_with_metadata_glm4_1v()
  # → hf_hub_download(sample_demo_1.mp4)，离线 pod collection 阶段崩，-k/-m 无法规避
  "--ignore=tests/models/multimodal/generation/test_common.py"
  # librosa 缺失（reference.md Landmine #17）
  "--ignore=tests/models/multimodal/generation/test_whisper.py"
  "--ignore=tests/models/multimodal/generation/test_phi4mm.py"
  # 模型未 staged + gguf 转换工具
  "--ignore=tests/models/multimodal/generation/test_multimodal_gguf.py"
  # mistral_common.audio 依赖
  "--ignore=tests/models/multimodal/generation/test_voxtral.py"
  "--ignore=tests/models/multimodal/generation/test_voxtral_realtime.py"
  # 视频资产未 vendor（红区无外网）
  "--ignore=tests/models/multimodal/generation/test_qwen2_5_vl.py"
  "--ignore=tests/models/multimodal/generation/test_qwen2_vl.py"
  "--ignore=tests/models/multimodal/generation/test_interleaved.py"
  # 音频资产从 S3 下载，红区无外网
  "--ignore=tests/models/multimodal/generation/test_ultravox.py"
  # trust_remote_code 动态代码下载失败
  "--ignore=tests/models/multimodal/generation/test_keye.py"
  # processing/ 在 mm_processing step 单独处理
  "--ignore=tests/models/multimodal/processing"
  # 注册表测试，遍历未 aliased 模型
  "--ignore=tests/models/multimodal/test_mapping.py"
  # Prithvi 模型未 aliased
  "--ignore=tests/models/multimodal/pooling/test_prithvi_mae.py"
  # module-level 读 examples/ jinja 模板，CI tar.gz 无 examples/
  "--ignore=tests/models/multimodal/pooling/test_llama_nemotron_vl.py"
  # VLM2Vec-Full trust_remote_code 下载失败
  "--ignore=tests/models/multimodal/pooling/test_phi3v.py"
)

# Step mm_processing —— 对位上游 Multi-Modal Processor（Step 6）：processor
# tensor schema 验证。注意 test_tensor_schema.py 顶部 `from .test_common import
# get_model_ids_to_test, get_text_token_prompts`，import 即执行 test_common 模块
# 体（含 module-level 视频下载），故本 step 依赖 [deps] 的 opencv + [assets] 的
# sample_demo_1.mp4 离线兜底；二者缺失时会在 collection 阶段崩（首跑观察，见
# aone 原 yaml「可能触发 Landmine #19 (VllmConfig NoneType)」注记）。
MM_PROCESSING_ARGS=(
  tests/models/multimodal/processing/test_tensor_schema.py
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
    # test_nemotron_parse.py::test_models（唯一 core_model，trust_remote_code=True）
    # 命中：aone_ci/scripts/ppu_model_aliases.json 键 nv-community/... →
    # /ppusw/datasets/checkpoints/LLM/NVIDIA/v1.1/NVIDIA-Nemotron-Parse-v1.1
    # （/ppusw → /nas_aisw）。测试用 HF id 是 nvidia/...（org 不同、指向同一份
    # 权重），故此处按测试实际 HF id 建 symlink。trust_remote_code 需 NAS 目录内
    # 含 modeling_*.py 等动态模块；缺失则该用例 fail（待入库补全 .py）。
    "nvidia/NVIDIA-Nemotron-Parse-v1.1":
        "/nas_aisw/datasets/checkpoints/LLM/NVIDIA/v1.1/NVIDIA-Nemotron-Parse-v1.1",
    # pooling/test_llava_next.py MODELS（core_model）
    # 命中：ppu_model_aliases.json royokong/e5-v →
    # /ppusw/datasets/checkpoints/LLM/e5/v1.0/e5-v（/ppusw → /nas_aisw）
    "royokong/e5-v": "/nas_aisw/datasets/checkpoints/LLM/e5/v1.0/e5-v",
    # 注：mm_processing (test_tensor_schema.py) 遍历 HF_EXAMPLE_MODELS 注册表里的
    # 多模态模型做 processor 验证，用 dummy_hf_overrides 只需 config/tokenizer；
    # 涉及模型众多且多数用例带 skip 守卫，此处不逐一枚举 —— 首跑观察 MISS 情况
    # 再按需补入（对齐 models-language 的 MISS 后补策略）。
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
# [assets] 离线兜底 vLLM 公共资产（图片 + 视频）
# ------------------------------------------------------------------------------
# 多模态测试用 vllm/assets 从 s3://vllm-public-assets（图片）与 HF dataset
# raushan-testing-hf/videos-test（视频 sample_demo_1.mp4）拉素材，红区无外网 →
# 需预置。两处 module-level 触点会在 collection 阶段就崩：
#   - test_nemotron_parse.py 顶部 IMAGE = ImageAsset("paper-11")...  → 需图片
#   - test_common.py（被 test_tensor_schema.py import）module-level 视频下载 → 需视频
# 兜底机制（对齐 aone_ci 侧 bootstrap，但运行时零依赖 aone_ci/，改为探测
# /nas_aisw 预置卷）：
#   - 图片 flat 落到 $VLLM_ASSETS_CACHE/vllm_public_assets/<file>
#     （见 vllm/assets/base.py:get_vllm_public_assets，命中即不下载）
#   - 视频 flat 落到 $VLLM_ASSETS_CACHE/video-example-data/sample_demo_1.mp4
#     （见 vllm/assets/video.py:download_video_asset，video_path 存在即不下载）
# 缺失时 WARN + 标「待入库」，不猜实际 NAS 路径硬填；首跑按实测补候选路径。
echo "========== [assets] bootstrap vLLM public assets (/nas_aisw) =========="
python3 - <<'PYEOF'
import os
import shutil

CACHE = os.environ.get("VLLM_ASSETS_CACHE") or os.path.expanduser(
    "~/.cache/vllm/assets")
IMG_DST = os.path.join(CACHE, "vllm_public_assets")
VID_DST = os.path.join(CACHE, "video-example-data")

# /nas_aisw 上 vllm 公共资产候选目录（首跑按实测收敛；红区清单未收录该类资产）
IMG_SRC_CANDS = [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/vision_model_images",
    "/nas_aisw/datasets/assets/vllm_public_assets",
]
VID_SRC_CANDS = [
    "/nas_aisw/datasets/vllm_public_assets",
    "/nas_aisw/datasets/vllm_assets/multimodal_asset",
    "/nas_aisw/datasets/assets/vllm_public_assets",
]

def _bootstrap(dst, cands, only=None):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for src in cands:
        if not os.path.isdir(src):
            continue
        for f in os.listdir(src):
            sp = os.path.join(src, f)
            if not os.path.isfile(sp):
                continue
            if only is not None and f not in only:
                continue
            dp = os.path.join(dst, f)
            if not os.path.exists(dp):
                shutil.copy(sp, dp)
                n += 1
    return n

img_n = _bootstrap(IMG_DST, IMG_SRC_CANDS)
vid_n = _bootstrap(VID_DST, VID_SRC_CANDS, only={"sample_demo_1.mp4"})
print(f"[assets] images bootstrapped={img_n} -> {IMG_DST}")
print(f"[assets] video bootstrapped={vid_n} -> {VID_DST}")
if not os.path.exists(os.path.join(IMG_DST, "paper-11.png")):
    print("[assets] WARN: paper-11.png absent (待入库) — "
          "test_nemotron_parse collection 会崩")
if not os.path.exists(os.path.join(VID_DST, "sample_demo_1.mp4")):
    print("[assets] WARN: sample_demo_1.mp4 absent (待入库) — "
          "test_common/test_tensor_schema collection 会崩")
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

root = ET.Element("testsuites", name="vLLM PPU Models Multimodal (GHA)")
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

lines = ["### Models Multimodal Test (PPU)", "",
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

# 复刻 aone single 段：single|all 均跑 mm_generation_dedicated + mm_processing
# 两个 step（本 area 无 multi 段，multi 模式已在上方 case 拒绝）。Aone pod 为
# 1-PPU，限 1 卡（CUDA_VISIBLE_DEVICES=0）保持单卡语义。
CUDA_VISIBLE_DEVICES=0 _run_step "mm_generation_dedicated" 1 "${MM_GENERATION_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 _run_step "mm_processing" 1 "${MM_PROCESSING_ARGS[@]}"

# ------------------------------------------------------------------------------
# [summary] 聚合退出码（sh 退出码 = 聚合 rc，保证 CI 信号不失真）
# ------------------------------------------------------------------------------
echo "========== [summary] steps:${STEP_LABELS_LIST} TOTAL_RC=${TOTAL_RC} =========="
exit "${TOTAL_RC}"
