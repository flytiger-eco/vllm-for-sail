#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that test paths referenced by the PPU area scripts still exist.

Background: the `--ignore=` / `--deselect` entries and the directly listed
test files in the three test-area-ppu-*.sh scripts were all copied by hand from
a point-in-time snapshot of `.buildkite/test_areas/` + `legacy_ci/ppu_extras/`.
After upstream renames or deletes a file:

- `--ignore=` pointing at a missing path: pytest silently ignores it, so the
  excluded tests are **not** actually run, with no warning at all (legacy CI
  already hit this with 4 stale entrypoints entries and 1 stale kernels entry).
- A directly listed test file that is missing: pytest reports an error, which
  is an explicit failure and less of a problem.

This check is purely static, needs no PPU device and runs on ubuntu-latest. Any
hit turns the PR red, forcing the exclusions to be maintained along with
upstream.

Usage: python3 scripts/ppu/check-exclusions.py [--quiet]
"""

from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT_GLOB = "scripts/ppu/test-area-ppu-*.sh"

# The two pytest flags that take a path; both --ignore=<path> and
# --ignore <path> forms must be supported.
# Regular expressions are avoided on purpose: the repo bans `import re` (it
# requires `regex`), while this script runs on the bare python3 of
# ubuntu-latest and must not pull in third-party dependencies.
PATH_FLAGS = ("--ignore", "--deselect")


def _tokens(line: str) -> list[str]:
    """Split into shell words, dropping comments, trailing backslash and quotes."""
    if " #" in line:
        line = line.split(" #", 1)[0]
    tokens = line.replace('"', " ").replace("'", " ").split()
    if tokens and tokens[-1] == "\\":
        tokens.pop()
    return tokens


def referenced_paths(script: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, repo-relative path)] with any pytest `::xxx` stripped."""
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
        # A selection entry on its own line (no flag): `tests/foo/test_bar.py`
        if not candidates and len(tokens) == 1 and tokens[0].startswith("tests/"):
            candidates.append(tokens[0])
        for candidate in candidates:
            path = candidate.split("::", 1)[0].strip().rstrip("/")
            # Variable interpolation (e.g. ${STUB_MODEL_DIR}) cannot be
            # resolved statically; skip it.
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
            "\n[check-exclusions] The following test paths do not exist on "
            "this branch: upstream renamed/deleted them without the "
            "exclusions being updated, so the excluded tests are not "
            "actually being run:",
            file=sys.stderr,
        )
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        print(
            "\nFix: update the selection and the --ignore/--deselect lists "
            "in the matching script against .buildkite/test_areas/<area>.yaml.",
            file=sys.stderr,
        )
        return 1
    if not quiet:
        print("[check-exclusions] OK - all referenced test paths exist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
