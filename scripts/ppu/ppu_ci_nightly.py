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
import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import regex as re

SUPPORTED_RELEASES = ("v0.26.0", "v0.27.0", "v0.28.0")
# Test paths an area script may reference although only some release branches carry
# them. Without an entry here the layout check reports the path as missing, which is
# the point: a dead pytest path silently changes what the area collects.
RELEASE_ONLY_PATHS = {
    "tests/v1/attention/test_indexer_native_next_n.py": ("v0.28.0",),
    "tests/v1/attention/test_mm_prefix.py": ("v0.28.0",),
}
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
    releases = RELEASE_ONLY_PATHS.get(relative)
    if releases is not None:
        return release is None or release in releases
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
        release_scopes: list[tuple[set[str], set[str]]] = []
        for line_number, line in enumerate(
            script.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if stripped.startswith(("if ", "elif ")) and stripped.endswith("; then"):
                comparison = r'"\$\{PPU_RELEASE:-\}"\s+(==|!=)\s+"(v[\d.]+)"'
                comparisons = re.findall(comparison, stripped)
                condition = re.sub(comparison, "release", stripped)
                accepted = remaining = set(SUPPORTED_RELEASES)
                # Check both branches of unknown conditions to avoid missing paths.
                if re.fullmatch(
                    r"(?:if|elif) \[\[ release(?: (?:\|\||&&) release)* \]\]; then",
                    condition,
                ) and not ("||" in condition and "&&" in condition):
                    combine = any if "||" in stripped else all
                    accepted = {
                        version
                        for version in SUPPORTED_RELEASES
                        if combine(
                            (version == value) == (op == "==")
                            for op, value in comparisons
                        )
                    }
                    remaining = set(SUPPORTED_RELEASES) - accepted
                if stripped.startswith("elif ") and release_scopes:
                    _, prior_remaining = release_scopes.pop()
                    accepted = accepted & prior_remaining
                    remaining = remaining & prior_remaining
                release_scopes.append((accepted, remaining))
            elif stripped.startswith("else") and release_scopes:
                release_scopes[-1] = (release_scopes[-1][1], set())
            elif stripped == "fi" and release_scopes:
                release_scopes.pop()
            declaration = re.match(
                r"(?:\w+_(?:ARGS|DESELECTS)|PYTEST_EXTRA)\+?=\(", stripped
            )
            if declaration:
                in_args = True
                stripped = stripped[declaration.end() :]
            if not in_args:
                continue
            lexer = shlex.shlex(stripped, posix=True, punctuation_chars="()")
            lexer.whitespace_split = True
            tokens = list(lexer)
            content = " ".join(tokens)
            if tokens and tokens[-1] == ")":
                in_args = False
            if release and any(release not in scope[0] for scope in release_scopes):
                continue
            for match in path_pattern.finditer(content):
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
    if args.areas.strip() and not requested:
        raise SelectionError("no area in manual selection")
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


def merge_pytest_args(original, extra):
    forwarded, keywords, markers, positional = [], [], [], []
    try:
        diagnostic = shlex.split(extra)
    except ValueError as error:
        raise SelectionError(f"invalid PYTEST_EXTRA_ARGS: {error}") from error
    for arguments in (original, diagnostic):
        tokens = iter(arguments)
        for token in tokens:
            if token == "--":
                positional.extend(tokens)
                break
            if token in ("-k", "-m"):
                expression = next(tokens, None)
                if expression is None or expression.startswith("-"):
                    raise SelectionError(f"{token} requires a filter expression")
            elif token.startswith(("-k", "-m")):
                expression = token[2:].removeprefix("=")
            else:
                forwarded.append(token)
                continue
            if expression.strip():
                (markers if token[:2] == "-m" else keywords).append(expression)
    for option, expressions in (("-m", markers), ("-k", keywords)):
        if expressions:
            merged = " and ".join(f"({expr})" for expr in expressions)
            forwarded.extend([option, merged])
    if positional:
        forwarded.extend(["--", *positional])
    return forwarded


def command_pytest(args):
    arguments = args.arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    command = [
        "pytest",
        *merge_pytest_args(arguments, os.environ.get("PYTEST_EXTRA_ARGS", "")),
    ]
    print(f"[pytest] {shlex.join(command)}", flush=True)
    os.execvp(command[0], command)


def prepare_cache(source, destination):
    """Isolate writable HF refs while reusing model snapshots on NAS."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source):
        raise ValueError("private cache must be outside the shared cache")
    destination.mkdir(parents=True, exist_ok=False)
    for repo in source.glob("models--*"):
        target = destination / repo.name
        for snapshot in (repo / "snapshots").glob("*"):
            if snapshot.is_dir():
                (target / "snapshots").mkdir(parents=True, exist_ok=True)
                (target / "snapshots" / snapshot.name).symlink_to(snapshot.resolve())
        if (repo / "refs").is_dir():
            shutil.copytree(repo / "refs", target / "refs")
        main = target / "refs/main"
        if (target / "snapshots/main").exists() and not main.exists():
            main.parent.mkdir(parents=True, exist_ok=True)
            main.write_text("main", encoding="utf-8")


def _deselects(path, *names):
    return tuple(f"--deselect={path}::{name}" for name in names)


EXPECTED_EMPTY_SELECTIONS = {
    ("v0.26.0", "spec_decode_ngram_suffix"): (
        "tests/v1/e2e/spec_decode/test_spec_decode.py",
        "-k",
        "(ngram or suffix) and not suffix_decoding_acceptance "
        "and not speculative_config1",
        *_deselects(
            "tests/v1/e2e/spec_decode/test_lora_with_spec_decode.py",
            "test_batch_inference_correctness[model_setup0]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/test_spec_decode.py",
            "test_dflash_acceptance_rates[False]",
            "test_dflash_acceptance_rates[True]",
            "test_dflash_correctness[False]",
            "test_dflash_correctness[True]",
            "test_draft_model_parallel_drafting",
            "test_draft_model_quantization[False-draft_quantized]",
            "test_draft_model_quantization[False-target_quantized]",
            "test_draft_model_quantization[True-draft_quantized]",
            "test_draft_model_quantization[True-target_quantized]",
            "test_draft_model_realistic_example",
            "test_dspark_correctness_and_acceptance_rate",
            "test_eagle_correctness_light[FLASH_ATTN-deepseek_eagle]",
            "test_eagle_correctness_medium[FLASH_ATTN-qwen3_eagle3]",
            "test_gemma4_dspark_correctness_and_acceptance_rate",
            "test_medusa_acceptance_rate",
            "test_mtp_correctness[deepseek]",
            "test_mtp_correctness[gemma4-e4b]",
            "test_mtp_correctness[qwen3_5-hybrid]",
            "test_ngram_and_suffix_correctness[speculative_config0]",
            "test_ngram_and_suffix_correctness[speculative_config1]",
            "test_ngram_gpu_default_with_async_scheduling[async]",
            "test_speculators_model_integration[llama3_eagle3_speculator]",
            "test_speculators_model_integration[qwen3_eagle3_speculator]",
            "test_suffix_decoding_acceptance",
        ),
    ),
    ("v0.27.0", "spec_decode_ngram_suffix"): (
        "tests/v1/e2e/spec_decode/ngram_suffix",
        "-k",
        "not suffix_decoding_acceptance and not speculative_config1",
        *_deselects(
            "tests/v1/e2e/spec_decode/draft_model/test_draft_model.py",
            "test_draft_model_parallel_drafting",
            "test_draft_model_quantization[False-draft_quantized]",
            "test_draft_model_quantization[False-target_quantized]",
            "test_draft_model_quantization[True-draft_quantized]",
            "test_draft_model_quantization[True-target_quantized]",
            "test_draft_model_realistic_example",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py",
            "test_eagle_correctness_light[FLASH_ATTN-deepseek_eagle]",
            "test_eagle_correctness_medium[FLASH_ATTN-qwen3_eagle3]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/mtp/test_mtp.py",
            "test_mtp_correctness[deepseek]",
            "test_mtp_correctness[gemma4-e4b]",
            "test_mtp_correctness[qwen3_5-hybrid]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/ngram_suffix/test_ngram_suffix.py",
            "test_ngram_and_suffix_correctness[speculative_config0]",
            "test_ngram_gpu_default_with_async_scheduling[async]",
        ),
    ),
    ("v0.28.0", "spec_decode_draft_model"): (
        "tests/v1/e2e/spec_decode/draft_model",
        "--ignore=tests/v1/e2e/spec_decode/draft_model/test_lora.py",
        "-k",
        "not tensor_parallelism and not engine_args and not no_sync",
        *_deselects(
            "tests/v1/e2e/spec_decode/eagle/test_eagle_correctness.py",
            "test_eagle_correctness_light[FLASH_ATTN-deepseek_eagle]",
            "test_eagle_correctness_medium[FLASH_ATTN-qwen3_vl_eagle3]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/mtp/test_mtp.py",
            "test_mtp_correctness[deepseek]",
            "test_mtp_correctness[qwen3_5-hybrid]",
            "test_mtp_correctness[gemma4-e4b]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/ngram_suffix/test_ngram_suffix.py",
            "test_ngram_and_suffix_correctness[speculative_config0]",
        ),
        *_deselects(
            "tests/v1/e2e/spec_decode/draft_model/test_draft_model.py",
            "test_draft_model_correctness[True-args0]",
            "test_draft_model_correctness[True-args1]",
            "test_draft_model_correctness[False-args0]",
            "test_draft_model_correctness[False-args1]",
            "test_draft_model_realistic_example",
            "test_draft_model_parallel_drafting",
            "test_draft_model_quantization[True-target_quantized]",
            "test_draft_model_quantization[True-draft_quantized]",
            "test_draft_model_quantization[False-target_quantized]",
            "test_draft_model_quantization[False-draft_quantized]",
            "test_draft_model_moe_backend_override",
            "test_draft_model_moe_backend_inherits_target",
            "test_draft_model_moe_backend_default_auto",
        ),
    ),
    ("v0.28.0", "test_qwen3_with_multi_loras"): (
        "tests/lora/test_qwen3_with_multi_loras.py",
        *_deselects(
            "tests/lora/test_lora_functions.py",
            "test_lora_functions_async",
            "test_lora_functions_sync",
        ),
        *_deselects(
            "tests/lora/test_worker.py",
            "test_worker_apply_lora",
        ),
        *_deselects(
            "tests/lora/test_lora_manager.py",
            "test_lru_cache_worker_adapter_manager[cuda:0]",
            "test_worker_adapter_manager[cuda:0]",
        ),
        *_deselects(
            "tests/lora/test_qwen3_unembed.py",
            "test_qwen3_unembed_lora",
        ),
        *_deselects(
            "tests/lora/test_gemma4_tp.py",
            "test_gemma4_lora",
        ),
        *_deselects(
            "tests/lora/test_qwen3_with_multi_loras.py",
            "test_multi_loras_with_tp_sync",
            "test_multiple_lora_requests",
            "test_load_inplace_offline_reload",
            "test_load_inplace_false_no_reload",
        ),
    ),
}


def normalize_selection(arguments):
    normalized = []
    tokens = iter(arguments)
    for token in tokens:
        if token == "--":
            normalized.append(token)
            normalized.extend(tokens)
            break
        if token in ("--ignore", "--deselect"):
            value = next(tokens, None)
            if value is None:
                raise SelectionError(f"{token} requires a path or node ID")
            token = f"{token}={value}"
        normalized.append(token)
    return merge_pytest_args(normalized, "")


def record_empty_skip(path, release, label, selection):
    """Mark only verified empty release/step/selection combinations as skipped."""
    expected = EXPECTED_EMPTY_SELECTIONS.get((release, label))
    if expected is None or any(
        os.environ.get(name, "").strip()
        for name in ("PYTEST_EXTRA_ARGS", "PYTEST_ADDOPTS")
    ):
        return 1
    if normalize_selection(selection) != normalize_selection(expected):
        return 1
    tree = ET.parse(path)
    suites = list(tree.iter("testsuite"))
    if not suites or any(
        int(suite.get(key, "0"))
        for suite in suites
        for key in ("tests", "failures", "errors")
    ):
        return 1
    reason = "All cases excluded by PPU filters; confirmed in run 42157224670"
    suite = ET.Element(
        "testsuite",
        name=label,
        tests="1",
        failures="0",
        errors="0",
        skipped="1",
        time="0",
    )
    case = ET.SubElement(suite, "testcase", name="ci_step_skipped", classname=label)
    ET.SubElement(case, "skipped", message=reason)
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)
    print(f"[step] {release}/{label}: SKIPPED ({reason})")
    return 0


def command_skip_empty(args):
    selection = args.selection
    if selection[:1] == ["--"]:
        selection = selection[1:]
    return record_empty_skip(args.path, args.release, args.label, selection)


def summarize_results(results_dir, exit_code):
    """Combine JUnit and process status; empty collections never pass."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "### Nightly PPU results",
        "",
        "| step | tests | failed | errors | skipped | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    failed = exit_code != 0
    total = [0, 0, 0, 0]
    try:
        suites = list(ET.parse(results_dir / "test.xml").iter("testsuite"))
        if not suites:
            raise ValueError("no test suites")
        for suite in suites:
            counts = [
                int(suite.get(key, "0"))
                for key in ("tests", "failures", "errors", "skipped")
            ]
            tests, failures, errors, skipped = counts
            if tests <= 0 or failures or errors:
                status = "FAILED"
                failed = True
            else:
                status = "SKIPPED" if tests == skipped else "PASSED"
            total = [a + b for a, b in zip(total, counts)]
            lines.append(
                f"| {suite.get('name', '?')} | {tests} | {failures} | "
                f"{errors} | {skipped} | {status} |"
            )
        raw_files = list((results_dir / "raw").rglob("*.xml"))
        if not raw_files:
            raise ValueError("missing raw JUnit files")
        for path in raw_files:
            raw_suites = list(ET.parse(path).iter("testsuite"))
            if not raw_suites or any(
                int(suite.get("tests", "0")) <= 0
                or int(suite.get("failures", "0"))
                or int(suite.get("errors", "0"))
                for suite in raw_suites
            ):
                raise ValueError(f"invalid raw JUnit: {path}")
        for log in (results_dir / "raw").rglob("*-shard*.log"):
            if not log.with_suffix(".xml").is_file():
                raise ValueError(f"missing shard JUnit: {log.with_suffix('.xml')}")
    except (OSError, ET.ParseError, ValueError) as error:
        lines.append(f"\nFAILED: missing or invalid JUnit: {error}")
        failed = True
    status = "FAILED" if failed else "SKIPPED" if total[0] == total[3] else "PASSED"
    lines.append(
        f"\nTotal: **{status}**; tests={total[0]}, skipped={total[3]}, "
        f"process exit code={exit_code}"
    )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(failed)


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

    pytest_parser = sub.add_parser("run-pytest", help="intersect diagnostic keywords")
    pytest_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    pytest_parser.set_defaults(func=command_pytest)

    cache_parser = sub.add_parser("prepare-cache", help="isolate HF cache metadata")
    cache_parser.add_argument("source", type=Path)
    cache_parser.add_argument("destination", type=Path)
    cache_parser.set_defaults(
        func=lambda args: prepare_cache(args.source, args.destination)
    )

    skip_parser = sub.add_parser("skip-empty", help="record a known empty step")
    skip_parser.add_argument("path", type=Path)
    skip_parser.add_argument("release", choices=SUPPORTED_RELEASES)
    skip_parser.add_argument("label")
    skip_parser.add_argument("selection", nargs=argparse.REMAINDER)
    skip_parser.set_defaults(func=command_skip_empty)

    summary_parser = sub.add_parser("summarize", help="validate and summarize JUnit")
    summary_parser.add_argument("results_dir", type=Path)
    summary_parser.add_argument("exit_code", type=int)
    summary_parser.set_defaults(
        func=lambda args: summarize_results(args.results_dir, args.exit_code)
    )

    list_parser = sub.add_parser("list", help="print the registry")
    list_parser.set_defaults(func=command_list)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        return args.func(args)
    except SelectionError as error:
        print(f"[{args.command}] ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
