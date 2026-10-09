#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""校验 PPU area 脚本里引用的测试路径仍然存在。

背景：11 个 test-area-ppu-*.sh 里共约 100 条 `--ignore=` / `--deselect` 以及
一批直接列出的测试文件，全部是从 `.buildkite/test_areas/` + `aone_ci/ppu_extras/`
的某一时刻快照手抄过来的。上游改名/删除文件后：

- `--ignore=` 指向不存在的路径：pytest 静默忽略，被排除的用例其实**没有**被跑，
  但也没有任何提示（Aone 侧已发生过 entrypoints 4 条、kernels 1 条失效）。
- 直接列出的测试文件不存在：pytest 直接报 error，属显式失败，问题较小。

本检查纯静态、不需要 PPU 设备，跑在 ubuntu-latest 上，命中即让 PR 转红，
迫使排除项跟着上游一起维护。

用法：python3 scripts/ppu/check-exclusions.py [--quiet]
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_GLOB = "scripts/ppu/test-area-ppu-*.sh"

# 取路径的两个 pytest 开关；--ignore=<path> 与 --ignore <path> 两种写法都要支持。
# 刻意不用正则：仓库禁用 `import re`（要求换 regex），而本脚本要在
# ubuntu-latest 的裸 python3 上跑，不能引入第三方依赖。
PATH_FLAGS = ("--ignore", "--deselect")


def _tokens(line: str) -> list[str]:
    """按 shell 词切分：剥掉行内注释、行尾续行符与引号。"""
    if " #" in line:
        line = line.split(" #", 1)[0]
    tokens = line.replace('"', " ").replace("'", " ").split()
    if tokens and tokens[-1] == "\\":
        tokens.pop()
    return tokens


def referenced_paths(script: pathlib.Path) -> list[tuple[int, str]]:
    """返回 [(行号, 仓库相对路径)]，已剥掉 pytest nodeid 的 ::xxx 部分。"""
    found: list[tuple[int, str]] = []
    for lineno, raw in enumerate(script.read_text().splitlines(), start=1):
        line = raw.strip()
        if line.startswith("#"):
            continue
        tokens = _tokens(line)
        candidates: list[str] = []
        for idx, token in enumerate(tokens):
            for flag in PATH_FLAGS:
                if token.startswith(f"{flag}="):
                    candidates.append(token[len(flag) + 1 :])
                elif token == flag and idx + 1 < len(tokens):
                    candidates.append(tokens[idx + 1])
        # 选集条目单独占一行（无开关）的形式：`tests/foo/test_bar.py`
        if not candidates and len(tokens) == 1 and tokens[0].startswith("tests/"):
            candidates.append(tokens[0])
        for candidate in candidates:
            path = candidate.split("::", 1)[0].strip().rstrip("/")
            # 变量插值（如 ${STUB_MODEL_DIR}）无法静态判定，跳过
            if not path.startswith("tests/") or "$" in path:
                continue
            found.append((lineno, path))
    return found


def main() -> int:
    quiet = "--quiet" in sys.argv
    scripts = sorted(REPO_ROOT.glob(SCRIPT_GLOB))
    if not scripts:
        print(f"ERROR: no script matched {SCRIPT_GLOB}", file=sys.stderr)
        return 2

    missing: list[str] = []
    checked = 0
    for script in scripts:
        for lineno, path in referenced_paths(script):
            checked += 1
            if not (REPO_ROOT / path).exists():
                rel = script.relative_to(REPO_ROOT)
                missing.append(f"{rel}:{lineno}: {path}")

    if not quiet:
        print(
            f"[check-exclusions] {len(scripts)} scripts, "
            f"{checked} test path references checked"
        )
    if missing:
        print(
            "\n[check-exclusions] 以下测试路径在当前分支不存在——"
            "上游改名/删除后排除项未同步，被排除的用例实际未被执行：",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        print(
            "\n修复方式：对照 .buildkite/test_areas/<area>.yaml 更新对应脚本的"
            "选集与 --ignore/--deselect 列表。",
            file=sys.stderr,
        )
        return 1
    if not quiet:
        print("[check-exclusions] OK — 所有引用的测试路径都存在")
    return 0


if __name__ == "__main__":
    sys.exit(main())
