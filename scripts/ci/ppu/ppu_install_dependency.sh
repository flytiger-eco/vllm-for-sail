#!/bin/bash
# ==============================================================================
# scripts/ci/ppu/ppu_install_dependency.sh — PPU CI 依赖安装（GitHub Actions）
# ------------------------------------------------------------------------------
# 调用方：.github/workflows/pr-test-ppu.yml，在 PPU 基础镜像容器内执行：
#   pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-...-vllm0.23.0-py312
#
# 对位：
#   - Aone CI 侧 wheel/依赖安装分散在 .aoneci/build-wheel-ppu.yaml +
#     aone_ci/scripts/test_area_ppu_lora.sh 的 [setup] 段
#   - sglang-for-sail scripts/ci/ppu/ppu_install_dependency.sh 的组织方式
#
# 核心原则（sglang 踩坑经验，见其脚本注释）：镜像预装的 PPU 栈（带
# +v0.1.0.ppu2.1.1 local version 的 torch/vllm/triton/flashinfer 等）一律
# 不动——blind `--force-reinstall` 会用同版本号的另一种构建覆盖镜像里的
# 版本，甚至降级。只安装镜像里真正缺失的包。
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

PIP_INSTALL="python3 -m pip install --no-cache-dir"
# SAIL SDK v2.1.1 PyPI source
PPU_PIP_INDEX="https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple"

# flytiger artifactory 凭证（GitHub Secrets 注入；镜像公开时可无）
if [[ -n "${PPU_ARTIFACTORY_USER:-}" && -n "${PPU_ARTIFACTORY_PASSWORD:-}" ]]; then
    echo "machine pkg.flytiger-eco.com login ${PPU_ARTIFACTORY_USER} password ${PPU_ARTIFACTORY_PASSWORD}" > ~/.netrc
    chmod 600 ~/.netrc
fi

# 容器内以 root 运行 git 需要（vllm 版本号由 setuptools-scm 从 git 推导时也依赖它）
git config --global --add safe.directory "${REPO_ROOT}"

# ------------------------------------------------------------------------------
# [diag] 环境盘点：先看清镜像里已有什么，再决定装什么
# ------------------------------------------------------------------------------
echo "========== [diag] preinstalled stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
python3 -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm NOT importable"
python3 -m pip list 2>/dev/null | grep -iE "^(vllm|torch|triton|flashinfer|deep.ep|deep.gemm|tilelang|flash.mla|flash.attn|transformers) " || true

# ------------------------------------------------------------------------------
# [deps] PPU 依赖：镜像预装优先，缺失才从 flytiger PyPI 补
# ------------------------------------------------------------------------------
# 版本来自 SAIL SDK v2.1.1 用户指南的 PyPI 安装命令。注意：如果镜像预装的
# 版本与下表不同（通常更新），以镜像为准，不降级。
_ensure_pip_pkg() {
    local pkg="$1" ver="$2" extra_args="${3:-}"
    local installed
    installed="$(python3 -c "import importlib.metadata as m; print(m.version('${pkg}'))" 2>/dev/null || true)"
    if [[ -n "${installed}" ]]; then
        echo "[deps] ${pkg} already installed: ${installed} — keep image build"
        return 0
    fi
    echo "[deps] installing ${pkg}==${ver}"
    # shellcheck disable=SC2086  # extra_args is intentionally word-split
    ${PIP_INSTALL} "${pkg}==${ver}" ${extra_args} -i "${PPU_PIP_INDEX}"
}

# vllm 本体：不带 --no-deps（需解析依赖树）；其余 PPU 组件按指南 --no-deps
_ensure_pip_pkg "vllm" "0.23.0"
_ensure_pip_pkg "flashinfer_python" "0.6.8.post1" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_ep" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "deep_gemm" "1.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "tilelang" "0.1.8" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_mla" "2.0.0" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn_3" "2.8.2" "--no-deps --force-reinstall"
_ensure_pip_pkg "flash_attn" "2.7.4.post1" "--no-deps --force-reinstall"

# ------------------------------------------------------------------------------
# [deps] pytest 测试依赖（对位 test_area_ppu_lora.sh 的 [setup] 段）
# ------------------------------------------------------------------------------
# pytest-asyncio: tests/conftest.py asyncio fixture 需要
# tblib: pytest 跨进程 traceback 序列化（spawn worker 报错时）
# pytest-shard: single 模式 --shard-id/--num-shards 需要
# pyyaml: 备用通用依赖（用例选集已内联在 run_lora_tests.sh，不解析 yaml）
echo "========== [deps] pytest toolchain =========="
${PIP_INSTALL} pytest pytest-asyncio tblib pytest-shard pyyaml -i "${PPU_PIP_INDEX}"

# ------------------------------------------------------------------------------
# [cext] 源码树补齐编译产物：镜像预装 vllm 的 C 扩展 → REPO_ROOT/vllm
# ------------------------------------------------------------------------------
# pytest 以 REPO_ROOT 为 rootdir，tests 包（tests/__init__.py）使 REPO_ROOT 被
# prepend 到 sys.path 最前 → import vllm 解析到源码树（纯 Python，无编译
# 产物）而非 site-packages 里镜像的完整构建。vllm/platforms/ppu.py 继承
# NvmlCudaPlatform，而 cuda.py 顶层 import vllm._C → conftest 一加载就
# ModuleNotFoundError: No module named 'vllm._C'。
# bring-up 期做法：把预装 wheel 的编译产物拷进源码树，形成“Python 层 =
# 分支源码 + C 扩展 = 镜像构建”的混合形态。ABI 前提：分支未改 csrc/
# Python 绑定接口（当前改动集中在 vllm/lora 纯 Python）。正式流程后续
# 切换为 wheel 构建（对位 Aone .aoneci/build-wheel-ppu.yaml），届时未段可删。
# 注：定位 site-packages 用 sysconfig 而非 import vllm —— cwd=REPO_ROOT 时
# import vllm 走的就是源码树，拿到的是错误答案。
echo "========== [cext] borrow compiled extensions from image vllm =========="
SP_SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
SP_VLLM_DIR="${SP_SITE}/vllm"
if [ ! -d "${SP_VLLM_DIR}" ]; then
    echo "[cext] ERROR: image vllm not found at ${SP_VLLM_DIR}" >&2
    exit 1
fi
cext_copied=0
# 递归复制镜像 vllm 包内所有编译产物并保持相对路径：顶层 _C/_moe_C/... 之外，
# vllm_flash_attn/ 子目录里还有 _vllm_fa2_C/_vllm_fa3_C（FA2/FA3 可用性探针，
# vllm/vllm_flash_attn/__init__.py 缺少它们会直接 raise ImportError）。
# 用 find 全量复制而非按文件名枚举：镜像增减扩展时本脚本无需同步改动。
while IFS= read -r f; do
    rel="${f#"${SP_VLLM_DIR}"/}"
    dest="${REPO_ROOT}/vllm/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "$f" "${dest}"
    echo "[cext] copied vllm/${rel}"
    cext_copied=1
done < <(find "${SP_VLLM_DIR}" -maxdepth 3 -type f -name '*.so')
if [ -f "${SP_VLLM_DIR}/_version.py" ]; then
    cp -f "${SP_VLLM_DIR}/_version.py" "${REPO_ROOT}/vllm/"
    echo "[cext] copied vllm/_version.py"
    cext_copied=1
fi
# auditwheel 打包的 wheel 会把依赖库放 vllm.libs/，RPATH 指向 $ORIGIN/../vllm.libs
if [ -d "${SP_VLLM_DIR}.libs" ]; then
    cp -rf "${SP_VLLM_DIR}.libs" "${REPO_ROOT}/"
    echo "[cext] copied vllm.libs/"
    cext_copied=1
fi
if [ "${cext_copied}" -eq 0 ]; then
    echo "[cext] ERROR: no compiled extensions found in ${SP_VLLM_DIR}" >&2
    exit 1
fi
# 对症验证：conftest 崩的是 vllm._C（cwd=REPO_ROOT，import 走源码树）
python3 -c "import vllm._C as c; print('[cext] vllm._C OK from source tree:', c.__file__)"
# 对症验证：本次挂的是 vllm.vllm_flash_attn（FA 扩展在子目录，顶层 glob 曾漏拷）
python3 -c "import vllm.vllm_flash_attn as fa; print('[cext] vllm.vllm_flash_attn OK: FA2=%s FA3=%s' % (fa.FA2_AVAILABLE, fa.FA3_AVAILABLE))"

echo "========== [verify] final stack =========="
python3 -c "import torch; print('torch', torch.__version__, 'device_count', torch.cuda.device_count())"
# 混合形态下 __version__ 来自镜像 wheel 的 _version.py（0.23.0+v0.2.0.ppu2.1.1），
# 不再是源码树的 'dev' fallback —— 这本身就是 [cext] 生效的信号
python3 -c "import vllm; print('vllm', vllm.__version__)"
echo "[deps] done"
