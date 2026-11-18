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
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

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
    Area("lora", nproc=4, run_timeout=120),
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


def stagger_lora(areas, release):
    """Place the four-card LoRA area at a release-specific matrix slot."""
    if release not in SUPPORTED_RELEASES or all(area.name != "lora" for area in areas):
        return areas
    lora = next(area for area in areas if area.name == "lora")
    ordered = [area for area in areas if area.name != "lora"]
    slot = {"v0.26.0": 4, "v0.27.0": 12}.get(release, len(ordered))
    ordered.insert(min(slot, len(ordered)), lora)
    return ordered


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
    # Keep summary commands independent of third-party packages on K8s.
    import regex as re

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
    if not requested:
        areas = stagger_lora(areas, args.release)
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


def _nightly_area_result(area_dir):
    try:
        suites = list(ET.parse(area_dir / "test.xml").iter("testsuite"))
        if not suites:
            raise ValueError("JUnit has no testsuite")
        totals = {
            key: sum(int(suite.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
    except (OSError, ValueError, ET.ParseError):
        return "FAILED", None
    try:
        exit_code = int((area_dir / "exit-code.txt").read_text().strip())
    except (OSError, ValueError):
        return "FAILED", totals
    if exit_code or totals["tests"] <= 0 or totals["failures"] or totals["errors"]:
        return "FAILED", totals
    return "PASSED", totals


_COUNT_KEYS = ("tests", "failures", "errors", "skipped")


def _attempt_root(results_root, run_id, attempt):
    return Path(results_root) / f"{run_id}-{attempt}" / "test"


def _resolve_area_dir(results_root, run_id, run_attempt, release, device, area):
    """Return the newest attempt directory that holds this area's results.

    The payload step claims the area directory before dispatch and the
    always-on validate step refreshes it, so an earlier attempt is only used
    for areas that were not rerun. A rerun area with missing or failed results
    stays FAILED instead of falling back to an older success.
    """
    for attempt in range(int(run_attempt), 0, -1):
        candidate = (
            _attempt_root(results_root, run_id, attempt) / release / device / area
        )
        if candidate.is_dir():
            return candidate, attempt
    return (
        _attempt_root(results_root, run_id, run_attempt) / release / device / area,
        int(run_attempt),
    )


def render_release_table(release, device, meta, areas, overall):
    """Render one release table from manifest-style area entries."""
    lines = [
        f"## {release} / {device}",
        "",
        f"Source SHA: `{meta.get('source_sha') or '<unavailable>'}`",
        f"CI SHA: `{meta.get('ci_sha') or '<unavailable>'}`",
        f"Wheel version: `{meta.get('wheel_version') or '<unavailable>'}`",
        f"Wheel SHA256: `{meta.get('wheel_sha256') or '<unavailable>'}`",
        "",
        "| Area | Tests | Failures | Errors | Skipped | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    totals = dict.fromkeys(_COUNT_KEYS, 0)
    for item in areas:
        counts = item.get("totals")
        values = {key: "-" for key in _COUNT_KEYS}
        if counts is not None:
            values = counts
            for key in _COUNT_KEYS:
                totals[key] += counts[key]
        lines.append(
            f"| {item['name']} | {values['tests']} | {values['failures']} | "
            f"{values['errors']} | {values['skipped']} | {item['result']} |"
        )
    lines.extend(
        [
            f"| **Total** | **{totals['tests']}** | **{totals['failures']}** | "
            f"**{totals['errors']}** | **{totals['skipped']}** | **{overall}** |",
            "",
        ]
    )
    return lines


def _manifest_valid(manifest, expected_areas):
    """Reject manifests with wrong coverage, fields, counts, or consistency."""
    if not isinstance(manifest, dict) or manifest.get("result") not in (
        "PASSED",
        "FAILED",
    ):
        return False
    areas = manifest.get("areas")
    if not isinstance(areas, list) or not areas:
        return False
    if {item.get("name") for item in areas} != set(expected_areas):
        return False
    for item in areas:
        if item.get("result") not in ("PASSED", "FAILED"):
            return False
        totals = item.get("totals")
        if totals is None:
            if item["result"] == "PASSED":
                return False
            continue
        if not isinstance(totals, dict) or any(
            not isinstance(totals.get(key), int) for key in _COUNT_KEYS
        ):
            return False
        clean = totals["tests"] > 0 and not totals["failures"] and not totals["errors"]
        if item["result"] == "PASSED" and not clean:
            return False
    return manifest["result"] == "FAILED" or all(
        item["result"] == "PASSED" for item in areas
    )


def _load_release_manifest(
    results_root, run_id, run_attempt, release, device, expected_areas
):
    """Read the newest release manifest; missing or invalid means FAILED."""
    current_attempt = int(run_attempt)
    for attempt in range(current_attempt, 0, -1):
        path = (
            _attempt_root(results_root, run_id, attempt)
            / release
            / device
            / "release-manifest.json"
        )
        if not path.is_file():
            # A claimed attempt without a manifest blocks older fallbacks.
            if path.parent.is_dir():
                return None
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            return manifest if _manifest_valid(manifest, expected_areas) else None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None
    return None


def command_nightly_summary(args):
    if args.output is None:
        raise SelectionError("summary-nightly requires --output or GITHUB_STEP_SUMMARY")
    needs = json.loads(os.environ["NEEDS_JSON"])
    selected = (
        needs.get("resolve-releases", {}).get("outputs", {}).get("selected", "")
    ).split(",")
    if any(release not in SUPPORTED_RELEASES for release in selected):
        raise SelectionError(f"invalid selected releases: {selected}")
    requested = [name for name in args.areas.replace(",", " ").split() if name]
    if args.areas.strip() and not requested:
        raise SelectionError("no area in manual selection")
    fallback_areas = select(requested)
    lines = [
        "# PPU nightly summary",
        "",
        "Detailed per-step results are available in the `ppu-results-*` artifacts.",
        "",
    ]
    for release in SUPPORTED_RELEASES:
        if release not in selected:
            continue
        manifest = _load_release_manifest(
            args.results_root,
            args.run_id,
            args.run_attempt,
            release,
            args.device,
            [area.name for area in fallback_areas],
        )
        if manifest is None:
            # Fail closed: without a trustworthy manifest every area is FAILED.
            meta = {"ci_sha": args.ci_sha}
            areas = [
                {"name": area.name, "result": "FAILED", "totals": None}
                for area in fallback_areas
            ]
            overall = "FAILED"
        else:
            meta = manifest
            areas = manifest["areas"]
            overall = manifest["result"]
        lines.extend(render_release_table(release, args.device, meta, areas, overall))
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return 0


def _release_area_manifest(area_dir, name, attempt):
    status, totals = _nightly_area_result(area_dir)
    required = ("summary.md", "test.xml", "exit-code.txt", "run.log", "job-info.txt")
    issues = [f"missing {path}" for path in required if not (area_dir / path).is_file()]
    # Unsharded steps log to run.log only, so raw logs must not be required.
    if not list(area_dir.glob("raw/**/*.xml")):
        issues.append("missing raw JUnit")
    if totals is None:
        issues.append("invalid test.xml")
    try:
        int((area_dir / "exit-code.txt").read_text().strip())
    except (OSError, ValueError):
        issues.append("invalid exit-code.txt")
    job_info_path = area_dir / "job-info.txt"
    if job_info_path.is_file():
        try:
            job_info = dict(
                line.split(": ", 1)
                for line in job_info_path.read_text().splitlines()
                if ": " in line
            )
            job_result = job_info.get("Result")
        except (OSError, UnicodeError):
            job_result = None
        if job_result not in ("PASSED", "FAILED"):
            issues.append("invalid job-info Result")
        elif job_result != status:
            issues.append(
                f"job-info Result {job_result} does not match test result {status}"
            )
    return {
        "name": name,
        "attempt": attempt,
        "present": area_dir.is_dir(),
        "result": "FAILED" if issues else status,
        "complete": not issues,
        "issues": issues,
        "totals": totals,
    }


def _matches_current_build(area_dir, args):
    """Earlier-attempt evidence is reusable only for the identical build."""
    try:
        info = dict(
            line.split(": ", 1)
            for line in (area_dir / "job-info.txt").read_text().splitlines()
            if ": " in line
        )
    except OSError:
        return False
    return (
        info.get("Source SHA") == args.source_sha
        and info.get("CI SHA") == args.ci_sha
        and info.get("Wheel version") == args.wheel_version
        and info.get("Wheel SHA256") == args.wheel_sha256
    )


def _github_api(path):
    request = urllib.request.Request(
        f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _retained_from_jobs(jobs, attempt_started_at, suffix):
    # Retained jobs carry the current run_attempt, so judge by execution time.
    retained = set()
    for job in jobs:
        name = job.get("name", "")
        completed_at = job.get("completed_at") or ""
        if (
            job.get("conclusion") == "success"
            and completed_at
            and completed_at < attempt_started_at
            and suffix in name
        ):
            retained.add(name.rsplit(suffix, 1)[1])
    return retained


def _retained_successful_areas(args):
    """Find area jobs retained from earlier attempts of this run."""
    repo = os.environ["GITHUB_REPOSITORY"]
    run = _github_api(f"/repos/{repo}/actions/runs/{args.run_id}")
    attempt_started_at = run["run_started_at"]
    suffix = f" / {args.release} / {args.device} / "
    retained = set()
    page = 1
    while True:
        payload = _github_api(
            f"/repos/{repo}/actions/runs/{args.run_id}"
            f"/jobs?filter=latest&per_page=100&page={page}"
        )
        jobs = payload["jobs"]
        retained |= _retained_from_jobs(jobs, attempt_started_at, suffix)
        if len(jobs) < 100:
            return retained
        page += 1


def command_release_summary(args):
    requested = [name for name in args.areas.replace(",", " ").split() if name]
    if args.areas.strip() and not requested:
        raise SelectionError("no area in manual selection")
    selected_areas = select(requested)
    current_attempt = int(args.run_attempt)
    retained = set()
    if current_attempt > 1:
        try:
            retained = _retained_successful_areas(args)
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(
                f"[summary-release] cannot confirm retained jobs: {error}; fail closed"
            )
    release_dir = (
        _attempt_root(args.results_root, args.run_id, args.run_attempt)
        / args.release
        / args.device
    )
    area_results = []
    for area in selected_areas:
        if current_attempt > 1 and area.name not in retained:
            area_dir = release_dir / area.name
            attempt = current_attempt
        else:
            area_dir, attempt = _resolve_area_dir(
                args.results_root,
                args.run_id,
                args.run_attempt,
                args.release,
                args.device,
                area.name,
            )
        target = release_dir / area.name
        if area_dir != target:
            # Materialize earlier-attempt evidence only for the same build;
            # anything else fails closed as missing evidence.
            if _matches_current_build(area_dir, args):
                shutil.copytree(area_dir, target)
            area_dir = target
        area_results.append(_release_area_manifest(area_dir, area.name, attempt))
    complete_count = sum(item["complete"] for item in area_results)
    passed = all(
        item["complete"] and item["result"] == "PASSED" for item in area_results
    )
    meta = {
        "source_sha": args.source_sha,
        "ci_sha": args.ci_sha,
        "wheel_version": args.wheel_version,
        "wheel_sha256": args.wheel_sha256,
    }
    lines = [
        "# PPU nightly summary",
        "",
        "Detailed per-step results are available in the `ppu-results-*` artifacts.",
        "",
    ]
    lines.extend(
        render_release_table(
            args.release,
            args.device,
            meta,
            area_results,
            "PASSED" if passed else "FAILED",
        )
    )
    content = "\n".join(lines)
    incomplete = [item["name"] for item in area_results if not item["complete"]]
    content += f"\nEvidence: **{complete_count}/{len(area_results)} complete**.\n"
    if incomplete:
        content += "Incomplete areas: " + ", ".join(incomplete) + "\n"
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "release": args.release,
        "device": args.device,
        "source_sha": args.source_sha,
        "ci_sha": args.ci_sha,
        "wheel_version": args.wheel_version,
        "wheel_sha256": args.wheel_sha256,
        "result": "PASSED" if passed else "FAILED",
        "expected_areas": len(area_results),
        "present_areas": sum(item["present"] for item in area_results),
        "complete_areas": complete_count,
        "passed_areas": sum(
            item["complete"] and item["result"] == "PASSED" for item in area_results
        ),
        "missing_areas": [item["name"] for item in area_results if not item["present"]],
        "incomplete_areas": incomplete,
        "areas": area_results,
    }
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "release-summary.md").write_text(content, encoding="utf-8")
    (release_dir / "release-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[summary-release] {args.release}/{args.device}: {manifest['result']}; "
        f"evidence={complete_count}/{len(area_results)}"
    )
    return int(not passed)


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
    select_parser.add_argument("--release", choices=SUPPORTED_RELEASES)
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

    nightly_summary_parser = sub.add_parser(
        "summary-nightly", help="render one summary table per release"
    )
    nightly_summary_parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="devops root holding {run_id}-{attempt}/test result trees",
    )
    nightly_summary_parser.add_argument("--areas", default=os.environ.get("AREAS", ""))
    nightly_summary_parser.add_argument("--device", required=True)
    nightly_summary_parser.add_argument("--ci-sha", required=True)
    nightly_summary_parser.add_argument("--run-id", required=True)
    nightly_summary_parser.add_argument("--run-attempt", required=True)
    nightly_summary_parser.add_argument(
        "--output", type=Path, default=os.environ.get("GITHUB_STEP_SUMMARY")
    )
    nightly_summary_parser.set_defaults(func=command_nightly_summary)

    release_summary_parser = sub.add_parser(
        "summary-release", help="write one release result bundle and evidence manifest"
    )
    release_summary_parser.add_argument(
        "--results-root",
        required=True,
        type=Path,
        help="devops root holding {run_id}-{attempt}/test result trees",
    )
    release_summary_parser.add_argument(
        "--release", required=True, choices=SUPPORTED_RELEASES
    )
    release_summary_parser.add_argument("--areas", default=os.environ.get("AREAS", ""))
    release_summary_parser.add_argument("--device", required=True)
    release_summary_parser.add_argument("--source-sha", required=True)
    release_summary_parser.add_argument("--ci-sha", required=True)
    release_summary_parser.add_argument("--wheel-version", required=True)
    release_summary_parser.add_argument("--wheel-sha256", required=True)
    release_summary_parser.add_argument("--run-id", required=True)
    release_summary_parser.add_argument("--run-attempt", required=True)
    release_summary_parser.set_defaults(func=command_release_summary)

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
