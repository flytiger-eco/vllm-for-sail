# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Daily PPU area registry for the maintained release branches.

The workflow supplies a release checkout through ``--source-root``. The same
24 areas run on every release every day; an explicit area list is only for
manual diagnosis. Test source paths are checked before building or scheduling
pods so a release layout change cannot turn into an empty successful test.
"""

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import regex as re

SUPPORTED_RELEASES = ("v0.26.0", "v0.27.0", "v0.28.0")
SCHEDULE_MARGIN_MINUTES = 25
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "ppu"
SCRIPT_PREFIX = "test-area-ppu-"


@dataclass(frozen=True)
class Area:
    name: str
    nproc: int
    run_timeout: int

    @property
    def script(self) -> str:
        return f"{SCRIPT_PREFIX}{self.name.replace('-', '_')}.sh"

    @property
    def timeout(self) -> int:
        return self.run_timeout + SCHEDULE_MARGIN_MINUTES

    def matrix_entry(self) -> dict:
        return {
            "area": self.name,
            "script": self.script,
            "nproc": self.nproc,
            "timeout": self.timeout,
            "run_timeout": self.run_timeout,
        }


AREAS = (
    Area("attention", nproc=1, run_timeout=35),
    Area("basic-correctness", nproc=2, run_timeout=90),
    Area("entrypoints", nproc=1, run_timeout=35),
    Area("entrypoints-llm", nproc=1, run_timeout=50),
    Area("lora", nproc=4, run_timeout=90),
    Area("model-executor", nproc=1, run_timeout=35),
    Area("models-basic", nproc=1, run_timeout=110),
    Area("samplers", nproc=1, run_timeout=35),
    Area("spec-decode", nproc=1, run_timeout=90),
    Area("engine", nproc=1, run_timeout=290),
    Area("kernels", nproc=1, run_timeout=350),
    Area("models-language", nproc=2, run_timeout=290),
    Area("benchmarks", nproc=1, run_timeout=90),
    Area("compile", nproc=2, run_timeout=90),
    Area("cuda", nproc=1, run_timeout=90),
    Area("distributed", nproc=2, run_timeout=90),
    Area("expert-parallelism", nproc=2, run_timeout=90),
    Area("misc", nproc=1, run_timeout=90),
    Area("model-runner-v2", nproc=2, run_timeout=90),
    Area("models-distributed", nproc=2, run_timeout=90),
    Area("models-multimodal", nproc=1, run_timeout=90),
    Area("pytorch", nproc=1, run_timeout=90),
    Area("quantization", nproc=1, run_timeout=90),
    Area("weight-loading", nproc=2, run_timeout=90),
)


class SelectionError(ValueError):
    pass


def by_name(name):
    for area in AREAS:
        if area.name == name:
            return area
    raise SelectionError(f"unknown area {name!r}")


def select(requested):
    names = [name for name in requested if name]
    if len(set(names)) != len(names):
        raise SelectionError("duplicate area in manual selection")
    return [by_name(name) for name in names] if names else list(AREAS)


def worst_case_minutes(areas):
    return sum(area.timeout for area in areas)


def format_minutes(minutes):
    hours, rest = divmod(minutes, 60)
    return f"{hours}h{rest:02d}m"


def emit_github_output(values, stream):
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in values.items():
            print(f"{key}={value}", file=stream)
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def path_supported_on_release(area, relative, release):
    if relative == "tests/v1/attention/test_indexer_native_next_n.py":
        return release is None or release == "v0.28.0"
    if release not in SUPPORTED_RELEASES or not relative.startswith(
        "tests/v1/e2e/spec_decode/"
    ):
        return True
    if area.name not in {"spec-decode", "model-runner-v2"}:
        return True
    basename = relative.rsplit("/", 1)[-1]
    legacy = basename in {
        "test_spec_decode.py",
        "test_lora_with_spec_decode.py",
    }
    return legacy if release == "v0.26.0" else not legacy


def check_source_layout(source_root, release=None):
    """Find dead pytest paths before collection can silently discard them."""
    root = Path(source_root).resolve()
    problems = []
    path_pattern = re.compile(r"(?<![\w./-])(tests/[\w./.*?\[\]-]+)")
    for area in AREAS:
        script = SCRIPT_DIR / area.script
        in_args = False
        for line_number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"_ARGS\+?=\($", stripped):
                in_args = True
                continue
            if in_args and stripped == ")":
                in_args = False
                continue
            if not in_args:
                continue
            for match in path_pattern.finditer(line):
                relative = match.group(1)
                if not path_supported_on_release(area, relative, release):
                    continue
                matches = (
                    glob.glob(str(root / relative))
                    if glob.has_magic(relative)
                    else [str(root / relative)]
                    if (root / relative).exists()
                    else []
                )
                if not matches:
                    problems.append(
                        f"{area.name}: {area.script}:{line_number}: "
                        f"missing {relative} for {release or 'this release'}"
                    )
    return problems


def check_registry(source_root=None, release=None):
    problems = []
    seen = set()
    for area in AREAS:
        if area.name in seen:
            problems.append(f"duplicate area {area.name}")
        seen.add(area.name)
        if area.nproc < 1:
            problems.append(f"{area.name}: nproc must be >= 1")
        if area.run_timeout < 1:
            problems.append(f"{area.name}: run_timeout must be >= 1")
        if not (SCRIPT_DIR / area.script).is_file():
            problems.append(
                f"{area.name}: missing "
                f"{SCRIPT_DIR.relative_to(REPO_ROOT)}/{area.script}"
            )
    if release is not None and release not in SUPPORTED_RELEASES:
        problems.append(f"unsupported release {release}")
    if source_root is not None:
        problems.extend(check_source_layout(source_root, release))
    return problems


def command_select(args):
    requested = [name for name in args.areas.replace(",", " ").split() if name]
    areas = select(requested)
    entries = [area.matrix_entry() for area in areas]
    matrix = json.dumps(entries, separators=(",", ":"))
    emit_github_output({"matrix": matrix, "count": str(len(entries))}, args.stream)
    names = ", ".join(area.name for area in areas)
    total = format_minutes(worst_case_minutes(areas))
    print(f"[select] daily areas={len(entries)}")
    print(f"[select] areas: {names}")
    print(f"[select] serial worst case: {total}")


def command_check(args):
    problems = check_registry(args.source_root, args.release)
    for problem in problems:
        print(f"[check] {problem}", file=args.stream)
    if problems:
        return 1
    print(f"[check] {len(AREAS)} daily areas; all scripts and test paths exist")
    return 0


def command_list(args):
    print(f"{'area':<22}{'nproc':<7}{'run':<6}{'job':<6}{'script'}")
    for area in AREAS:
        print(
            f"{area.name:<22}{area.nproc:<7}"
            f"{area.run_timeout:<6}{area.timeout:<6}{area.script}",
            file=args.stream,
        )
    print(
        f"[list] daily: {len(AREAS)} areas, serial worst case "
        f"{format_minutes(worst_case_minutes(AREAS))}",
        file=args.stream,
    )
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stream", default=sys.stdout, type=argparse.FileType("w"))
    sub = parser.add_subparsers(dest="command", required=True)

    select_parser = sub.add_parser("select", help="emit the GitHub Actions matrix")
    select_parser.add_argument(
        "--areas",
        default=os.environ.get("AREAS", ""),
        help="comma or space separated names; empty means all daily areas",
    )
    select_parser.set_defaults(func=command_select)

    check_parser = sub.add_parser("check", help="validate registry and source layout")
    check_parser.add_argument("--source-root", type=Path)
    check_parser.add_argument("--release", choices=SUPPORTED_RELEASES)
    check_parser.set_defaults(func=command_check)

    list_parser = sub.add_parser("list", help="print the registry")
    list_parser.set_defaults(func=command_list)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        return args.func(args)
    except SelectionError as error:
        print(f"[select] ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
