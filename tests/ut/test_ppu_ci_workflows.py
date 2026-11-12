# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""CPU-only contracts for the on-demand PPU CI workflows."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
import yaml

from scripts.ppu import ppu_ci_nightly as nightly

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"
DEFAULT_BRANCH = "${{ github.event.repository.default_branch }}"
HEAD_SHA = "${{ inputs.pr_head_sha }}"


def workflow(name: str):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def checkouts(job):
    return [
        step
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/checkout@")
    ]


def test_comment_controller_runs_trusted_default_branch_code():
    job = workflow("run-ci-command.yml")["jobs"]["command"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["concurrency"]["queue"] == "max"
    assert job["permissions"] == {
        "contents": "read",
        "actions": "write",
        "issues": "write",
        "pull-requests": "write",
        "statuses": "write",
    }
    checkout = checkouts(job)[0]
    assert checkout["with"] == {
        "ref": DEFAULT_BRANCH,
        "persist-credentials": False,
    }
    command = next(step for step in job["steps"] if "run" in step)["run"]
    assert command == "python scripts/ci/ppu_ci_command.py command"
    assert "BUILDKITE" not in str(job)


def test_ready_notification_uses_github_hosted_control_plane():
    config = workflow("notify-ci-authorized.yml")
    job = config["jobs"]["notify"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "ready-run-all-tests" in job["if"]
    checkout = checkouts(job)[0]
    assert checkout["with"]["ref"] == DEFAULT_BRANCH
    assert checkout["with"]["persist-credentials"] is False
    command = next(step for step in job["steps"] if "run" in step)["run"]
    assert command == "python scripts/ci/ppu_ci_command.py notify"


def test_executor_is_manual_and_serialized_per_pr():
    config = workflow("ppu-pr-ci.yml")
    assert set(config[True]) == {"workflow_dispatch"}
    assert set(config[True]["workflow_dispatch"]["inputs"]) == {
        "pr_number",
        "head_sha",
        "base_ref",
        "requester",
        "comment_id",
    }
    assert config["concurrency"] == {
        "group": "ppu-pr-ci-${{ inputs.pr_number }}",
        "cancel-in-progress": False,
        "queue": "max",
    }
    assert set(config["jobs"]) == {
        "validate",
        "build",
        "attention",
        "samplers",
        "benchmarks",
        "report",
    }


def test_executor_control_jobs_only_run_default_branch_code():
    config = workflow("ppu-pr-ci.yml")
    guard = (
        "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)"
    )
    for job_id in ("validate", "report"):
        job = config["jobs"][job_id]
        assert job["runs-on"] == "ubuntu-latest"
        checkout = checkouts(job)[0]
        assert checkout["with"]["ref"] == "${{ github.sha }}"
        assert checkout["with"]["persist-credentials"] is False
        if job_id == "validate":
            assert job["if"] == guard
        else:
            assert job["if"] == "always()"
            assert all(guard in step.get("if", guard) for step in job["steps"])
    assert config["jobs"]["report"]["needs"] == [
        "validate",
        "build",
        "attention",
        "samplers",
        "benchmarks",
    ]


def test_executor_calls_read_only_pr_workflows_without_write_permissions():
    jobs = workflow("ppu-pr-ci.yml")["jobs"]
    assert jobs["build"]["uses"] == "./.github/workflows/ppu-build-pr.yml"
    for area in ("attention", "samplers", "benchmarks"):
        job = jobs[area]
        assert job["uses"] == f"./.github/workflows/ppu-{area}-pr.yml"
        assert job["with"]["pr_head_sha"] == "${{ inputs.head_sha }}"
        assert job["permissions"] == {"contents": "read", "actions": "read"}
        assert job["secrets"] == {
            "PPU_ARTIFACTORY_USER": "${{ secrets.PPU_ARTIFACTORY_USER }}",
            "PPU_ARTIFACTORY_PASSWORD": "${{ secrets.PPU_ARTIFACTORY_PASSWORD }}",
        }


def test_build_checks_out_exact_head_and_names_retry_artifact():
    config = workflow("ppu-build-pr.yml")
    assert set(config[True]) == {"workflow_call"}
    assert set(config[True]["workflow_call"]["inputs"]) == {
        "pr_head_sha",
        "base_ref",
    }
    job = config["jobs"]["build-wheel"]
    assert job["name"] == "PPU wheel build"
    assert "github.event_name == 'workflow_dispatch'" in job["if"]
    assert "github.event.repository.default_branch" in job["if"]
    checkout = checkouts(job)[0]
    assert checkout["with"]["ref"] == HEAD_SHA
    assert checkout["with"]["persist-credentials"] is False
    assert any(step.get("name") == "Verify checkout SHA" for step in job["steps"])
    text = (WORKFLOWS / "ppu-build-pr.yml").read_text()
    assert 'TARGET_VERSION="${TARGET_VERSION#v}"' in text
    assert '"${{ steps.version.outputs.target_version }}"' in text
    assert "artifact_name=ppu-wheels-${{ github.run_id }}" in text
    upload = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["name"] == "${{ steps.version.outputs.artifact_name }}"
    assert upload["with"]["overwrite"] is True
    assert '"${{ inputs.base_ref }}"' in text


def test_area_jobs_have_read_only_tokens_and_exact_sha_checkout():
    for area in ("attention", "samplers", "benchmarks"):
        config = workflow(f"ppu-{area}-pr.yml")
        assert config["permissions"] == {"contents": "read", "actions": "read"}
        job = next(iter(config["jobs"].values()))
        assert "github.event_name == 'workflow_dispatch'" in job["if"]
        assert "github.event.repository.default_branch" in job["if"]
        assert job["runs-on"] == "k8s-runner-group-cpu-flytiger"
        checkout = checkouts(job)[0]
        assert checkout["with"]["ref"] == HEAD_SHA
        assert checkout["with"]["persist-credentials"] is False
        assert any(step.get("name") == "Verify checkout SHA" for step in job["steps"])
        action = next(
            step["uses"]
            for step in job["steps"]
            if step.get("uses", "").startswith("flytiger-eco/ppu-distributed-action@")
        )
        assert action.endswith("@06c8ee7cadaae2086db4977e571b19f461653f16")


def test_area_scripts_force_offline_and_keep_nas_cache_discovery():
    for area in nightly.AREAS:
        text = (nightly.SCRIPT_DIR / area.script).read_text()
        assert "PPU_TEST_ONLINE" not in text
        assert "export HF_HUB_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1" in text
        assert "/nas_aisw/datasets/hf_cache/hub" in text


def test_lightweight_checks_read_live_pr_and_treat_content_as_data_only():
    config = workflow("pr-light-checks.yml")
    assert set(config[True]) == {"pull_request_target"}
    assert "edited" in config[True]["pull_request_target"]["types"]
    jobs = config["jobs"]
    assert set(jobs) == {"source-check", "secret-scan"}
    source = jobs["source-check"]
    assert source["permissions"] == {"pull-requests": "read"}
    assert source["outputs"] == {
        "eligible": "${{ steps.source.outputs.eligible }}",
        "repository": "${{ steps.source.outputs.repository }}",
        "head_sha": "${{ steps.source.outputs.head_sha }}",
        "base_sha": "${{ steps.source.outputs.base_sha }}",
        "base_ref": "${{ steps.source.outputs.base_ref }}",
    }
    source_step = source["steps"][0]
    assert source_step["uses"].startswith("actions/github-script@")
    assert "github.rest.pulls.get" in source_step["with"]["script"]
    checkout = checkouts(jobs["secret-scan"])[0]
    assert checkout["with"]["repository"] == (
        "${{ needs.source-check.outputs.repository }}"
    )
    assert checkout["with"]["ref"] == "${{ needs.source-check.outputs.head_sha }}"
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["allow-unsafe-pr-checkout"] is True
    assert jobs["secret-scan"]["permissions"] == {"contents": "read"}
    fetch = next(
        step
        for step in jobs["secret-scan"]["steps"]
        if step.get("name") == "Fetch the target branch commit"
    )
    assert "git fetch --no-tags" in fetch["run"]
    assert "--depth" not in fetch["run"]
    assert "BASE_SHA" in fetch["run"]
    trufflehog = next(
        step["uses"]
        for step in jobs["secret-scan"]["steps"]
        if step.get("uses", "").startswith("trufflesecurity/trufflehog@")
    )
    assert trufflehog == (
        "trufflesecurity/trufflehog@e7a8d17d2f726cf414310a625b55cca7dc9c898e"
    )


def test_lifecycle_controller_runs_only_default_branch_code():
    config = workflow("cancel-obsolete-ppu-ci.yml")
    assert set(config[True]) == {
        "pull_request_target",
        "schedule",
        "workflow_dispatch",
    }
    job = config["jobs"]["lifecycle"]
    assert job["runs-on"] == "ubuntu-latest"
    assert config["concurrency"]["queue"] == "max"
    checkout = checkouts(job)[0]
    assert checkout["with"] == {
        "ref": DEFAULT_BRANCH,
        "persist-credentials": False,
    }
    command = next(step for step in job["steps"] if "run" in step)["run"]
    assert command == "python -m scripts.ci.ppu_ci_lifecycle"


def test_credential_bearing_workflows_pin_actions_images_and_secrets():
    forbidden = ("@v4", "@main", ":latest")
    for area in ("attention", "samplers", "benchmarks"):
        config = workflow(f"ppu-{area}-pr.yml")
        assert set(config[True]["workflow_call"]["secrets"]) == {
            "PPU_ARTIFACTORY_USER",
            "PPU_ARTIFACTORY_PASSWORD",
        }
        text = (WORKFLOWS / f"ppu-{area}-pr.yml").read_text()
        assert not any(token in text for token in forbidden)
    build = (WORKFLOWS / "ppu-build-pr.yml").read_text()
    assert not any(token in build for token in forbidden)


def test_legacy_automatic_pipeline_is_cancelled_before_heavy_jobs():
    config = workflow("cancel-legacy-ppu-ci.yml")
    assert config[True]["workflow_run"] == {
        "workflows": ["CI Pipeline"],
        "types": ["requested", "in_progress"],
    }
    job = config["jobs"]["cancel"]
    assert job["permissions"] if "permissions" in job else config["permissions"]
    assert "pull_request_review" in job["if"]
    assert "cancelWorkflowRun" in str(job)


def test_legacy_pipeline_is_disabled_after_default_branch_merge():
    config = workflow("disable-legacy-ppu-ci.yml")
    assert set(config[True]) == {"push", "workflow_dispatch"}
    assert config[True]["push"]["branches"] == ["main"]
    job = config["jobs"]["disable"]
    assert config["permissions"] == {"actions": "write"}
    script = job["steps"][0]["with"]["script"]
    assert "disableWorkflow" in script
    assert "workflow_id: 349122804" in script


def test_pr_commenting_jobs_use_job_scoped_github_token():
    expected = "${{ github.token }}"
    cases = [
        ("run-ci-command.yml", "command", "Handle PPU CI command"),
        ("notify-ci-authorized.yml", "notify", "Explain PPU CI commands"),
        ("ppu-pr-ci.yml", "report", "Report PPU CI result"),
    ]
    for workflow_name, job_id, step_name in cases:
        job = workflow(workflow_name)["jobs"][job_id]
        assert job["permissions"]["issues"] == "write"
        assert job["permissions"]["pull-requests"] == "write"
        step = next(item for item in job["steps"] if item.get("name") == step_name)
        assert step["env"]["GITHUB_TOKEN"] == expected

    command = workflow("run-ci-command.yml")["jobs"]["command"]
    reporter = next(
        item
        for item in command["steps"]
        if item.get("name") == "Report unexpected controller failure"
    )
    assert reporter["with"]["github-token"] == expected


def test_non_commenting_control_jobs_keep_configured_pat_fallback():
    expected = "${{ secrets.CI_BRANCH_UPDATE_TOKEN || github.token }}"
    cases = [
        ("ppu-pr-ci.yml", "validate", "Validate PPU CI request"),
        (
            "cancel-obsolete-ppu-ci.yml",
            "lifecycle",
            "Cancel ineligible or superseded PPU runs",
        ),
    ]
    for workflow_name, job_id, step_name in cases:
        job = workflow(workflow_name)["jobs"][job_id]
        step = next(item for item in job["steps"] if item.get("name") == step_name)
        assert step["env"]["GITHUB_TOKEN"] == expected


@pytest.mark.parametrize(
    "original,extra,expected",
    [
        (["tests/unit", "-x"], "", ["tests/unit", "-x"]),
        (
            ["-k", "not cpu", "-m", "single"],
            '-k "cuda or cpu" -x --ignore="tests/unused file.py"',
            [
                "-x",
                "--ignore=tests/unused file.py",
                "-m",
                "(single)",
                "-k",
                "(not cpu) and (cuda or cpu)",
            ],
        ),
        (["-k", "not cpu"], '-k ""', ["-k", "(not cpu)"]),
        (
            ["-knot cpu"],
            '-k="cuda or cpu" -k "not slow"',
            ["-k", "(not cpu) and (cuda or cpu) and (not slow)"],
        ),
        (["--", "-keep.py"], "-k cuda", ["-k", "(cuda)", "--", "-keep.py"]),
        (
            ["tests/unit", "-mnot gpu", "-k", "not cpu"],
            '-m "cuda or cpu" -m "not slow" -k marlin',
            [
                "tests/unit",
                "-m",
                "(not gpu) and (cuda or cpu) and (not slow)",
                "-k",
                "(not cpu) and (marlin)",
            ],
        ),
        (["-m", "single"], '-m ""', ["-m", "(single)"]),
    ],
)
def test_nightly_diagnostic_keywords_preserve_constraints(original, extra, expected):
    assert nightly.merge_pytest_args(original, extra) == expected


@pytest.mark.parametrize("extra", ["-k", "-k -x", '-k "unterminated', "-m", "-m -x"])
def test_nightly_invalid_diagnostic_args_fail_before_pytest(monkeypatch, capsys, extra):
    monkeypatch.setenv("PYTEST_EXTRA_ARGS", extra)
    monkeypatch.setattr(os, "execvp", lambda *args: pytest.fail("unexpected execution"))
    assert nightly.main(["run-pytest", "--", "-k", "not cpu"]) == 2
    assert "ERROR:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra,expected_count",
    [('-k "cuda or cpu"', 1), ("-k cpu", 0), ("-m single", 1), ("-m 'not single'", 0)],
)
def test_nightly_diagnostic_collection_is_intersection(tmp_path, extra, expected_count):
    (tmp_path / "test_selection.py").write_text(
        "import pytest\n"
        "@pytest.mark.single\ndef test_cuda(): pass\n"
        "@pytest.mark.single\ndef test_cpu(): pass\n"
        "@pytest.mark.single\ndef test_cuda_blocked(): pass\n"
        "def test_cuda_multi(): pass\n"
    )
    (tmp_path / "test_ignored.py").write_text("def test_cuda(): pass\n")
    junit = tmp_path / "result.xml"
    result = subprocess.run(
        [
            sys.executable,
            str(nightly.SCRIPT_DIR / "ppu_ci_nightly.py"),
            "run-pytest",
            "--",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "markers=single",
            "--confcutdir",
            str(tmp_path),
            "--ignore=test_ignored.py",
            "--deselect=test_selection.py::test_cuda_blocked",
            "-m",
            "single",
            "-k",
            "not cpu",
            f"--junit-xml={junit}",
            ".",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
            "PYTEST_EXTRA_ARGS": extra,
            "PYTEST_ADDOPTS": "",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == (0 if expected_count else 5), (
        result.stdout + result.stderr
    )
    cases = list(ET.parse(junit).iter("testcase"))
    assert [case.get("name") for case in cases] == (
        ["test_cuda"] if expected_count else []
    )


@pytest.mark.parametrize("area", nightly.AREAS, ids=lambda area: area.name)
def test_nightly_area_routes_quoted_diagnostics_through_shared_runner(area):
    text = (nightly.SCRIPT_DIR / area.script).read_text()
    wrapper = "pytest() {" + text.split("pytest() {", 1)[1].split("\n}", 1)[0] + "\n}"
    assert "read -ra PYTEST_EXTRA" not in text
    result = subprocess.run(
        ["bash"],
        input=f'''
set -euo pipefail
REPO_ROOT="{ROOT}"
python3() {{ printf '%s\\n' "$@"; }}
{wrapper}
pytest -k "not cpu" --junit-xml="result with spaces.xml"
''',
        env={**os.environ, "PYTEST_EXTRA_ARGS": '-k "cuda or cpu"'},
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(nightly.SCRIPT_DIR / "ppu_ci_nightly.py"),
        "run-pytest",
        "--",
        "-k",
        "not cpu",
        "--junit-xml=result with spaces.xml",
    ]


@pytest.mark.parametrize(
    "areas,count",
    [("", 24), (" ", 24), ("attention,distributed", 2), (",,", None), (" , , ", None)],
)
def test_nightly_manual_selection_does_not_expand_empty_entries(
    monkeypatch, capsys, areas, count
):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    rc = nightly.main(["select", "--areas", areas])
    output = capsys.readouterr()
    if count is None:
        assert rc == 2
        assert "no area in manual selection" in output.err
        assert "matrix=" not in output.out
    else:
        assert not rc
        assert f"count={count}\n" in output.out


@pytest.mark.parametrize("release", nightly.SUPPORTED_RELEASES)
@pytest.mark.parametrize(
    "declaration",
    [
        "WL_ARGS=(tests/selected.py)",
        "TEST_ARGS=(\n tests/selected.py\n)",
        "TEST_ARGS=(\n -m 'distributed(num_gpus=2)'\n tests/selected.py\n)",
        'PYTEST_EXTRA+=(\n --deselect "tests/selected.py::test_case[x]"\n)',
        'MISC_V1_INFERENCE_DESELECTS=("tests/selected.py::test_case[x]")',
    ],
)
def test_nightly_rejects_missing_paths_in_supported_arrays(
    tmp_path, monkeypatch, release, declaration
):
    area = nightly.Area("weight-loading", 2, 90)
    monkeypatch.setattr(nightly, "AREAS", (area,))
    monkeypatch.setattr(nightly, "SCRIPT_DIR", tmp_path)
    (tmp_path / area.script).write_text(
        declaration + " # tests/comment.py\necho tests/not-selected.py\n"
    )
    root = tmp_path / "source"
    assert any(
        "missing tests/selected.py" in problem
        for problem in nightly.check_source_layout(root, release)
    )
    (root / "tests").mkdir(parents=True)
    (root / "tests/selected.py").touch()
    assert nightly.check_source_layout(root, release) == []


@pytest.mark.parametrize("release", nightly.SUPPORTED_RELEASES)
def test_nightly_checks_release_only_paths_on_the_matching_release(
    tmp_path, monkeypatch, release
):
    area = nightly.Area("attention", 1, 35)
    monkeypatch.setattr(nightly, "AREAS", (area,))
    monkeypatch.setattr(nightly, "SCRIPT_DIR", tmp_path)
    relative = "tests/v1/attention/test_mm_prefix.py"
    (tmp_path / area.script).write_text(f"ATTN_ARGS=(--ignore={relative})\n")
    assert bool(nightly.check_source_layout(tmp_path, release)) == (
        release == "v0.28.0"
    )
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    assert nightly.check_source_layout(tmp_path, release) == []


@pytest.mark.parametrize("release", nightly.SUPPORTED_RELEASES)
def test_nightly_layout_checks_release_branches_and_unknown_conditions(
    tmp_path, monkeypatch, release
):
    area = nightly.Area("models-basic", 1, 35)
    monkeypatch.setattr(nightly, "AREAS", (area,))
    monkeypatch.setattr(nightly, "SCRIPT_DIR", tmp_path)
    (tmp_path / area.script).write_text("""
if [[ "${PPU_RELEASE:-}" == "v0.26.0" ]]; then
  PYTEST_EXTRA+=(--ignore=tests/v0.26.0.py)
elif [[ "${PPU_RELEASE:-}" == "v0.27.0" ]]; then
  PYTEST_EXTRA+=(--ignore=tests/v0.27.0.py)
else
  if [[ -n "${OPTIONAL:-}" ]]; then
    PYTEST_EXTRA+=(--ignore=tests/v0.28.0.py)
  fi
fi
if [[ "${PPU_RELEASE:-}" == "v0.28.0" || -n "${OPTIONAL:-}" ]]; then
  TEST_ARGS=(tests/optional.py)
else
  TEST_ARGS=(tests/fallback.py)
fi
""")
    problems = nightly.check_source_layout(tmp_path, release)
    assert len(problems) == 3
    for relative in (f"tests/{release}.py", "tests/optional.py", "tests/fallback.py"):
        assert any(f"missing {relative}" in problem for problem in problems)


def test_nightly_parallel_caches_keep_refs_private_and_reuse_weights(tmp_path):
    source = tmp_path / "shared"
    repo = "models--org--model"
    snapshot = source / repo / "snapshots/main"
    snapshot.mkdir(parents=True)
    (snapshot / "weights").write_text("weights")
    refs = source / repo / "refs/main"
    refs.parent.mkdir()
    refs.write_text("main")
    destinations = [tmp_path / "job1", tmp_path / "job2"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda dest: nightly.prepare_cache(source, dest), destinations))
    for i, destination in enumerate(destinations):
        local_repo = destination / repo
        assert (local_repo / "snapshots/main").resolve() == snapshot
        assert not (local_repo / "refs/main").is_symlink()
        (local_repo / "refs/main").write_text(f"job{i}")
        cold = destination / "models--org--cold/snapshots"
        cold.mkdir(parents=True)
        (cold / "main").symlink_to(snapshot)
    assert refs.read_text() == "main"
    assert not (source / "models--org--cold").exists()


def test_nightly_private_cache_repairs_an_unpublished_main_ref(tmp_path):
    source, destination = tmp_path / "shared", tmp_path / "private"
    (source / "models--org--model/snapshots/main").mkdir(parents=True)
    nightly.prepare_cache(source, destination)
    assert (destination / "models--org--model/refs/main").read_text() == "main"
    assert not (source / "models--org--model/refs").exists()
    with pytest.raises(ValueError):
        nightly.prepare_cache(source, source / "private")


@pytest.mark.parametrize(
    "xml,exit_code,expected",
    [
        ('<testsuite tests="1"/>', 0, "PASSED"),
        ('<testsuite tests="1"/>', 7, "FAILED"),
        ('<testsuite tests="0"/>', 0, "FAILED"),
        ('<testsuite tests="1" failures="1"/>', 0, "FAILED"),
        ('<testsuite tests="1" errors="1"/>', 0, "FAILED"),
        ('<testsuite tests="1" skipped="1"/>', 0, "SKIPPED"),
        ("<testsuites/>", 0, "FAILED"),
        ("broken", 0, "FAILED"),
        (None, 0, "FAILED"),
    ],
)
def test_nightly_summary_fails_closed(tmp_path, xml, exit_code, expected):
    if xml is not None:
        (tmp_path / "test.xml").write_text(xml)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/step.xml").write_text('<testsuite tests="1"/>')
    assert nightly.summarize_results(tmp_path, exit_code) == int(expected == "FAILED")
    assert f"Total: **{expected}**" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("raw_state", ["missing", "empty", "log-only"])
def test_nightly_summary_requires_raw_junit(tmp_path, raw_state):
    (tmp_path / "test.xml").write_text('<testsuite tests="1"/>')
    raw = tmp_path / "raw"
    if raw_state != "missing":
        raw.mkdir()
    if raw_state == "log-only":
        (raw / "run.log").write_text("test output")
    assert nightly.summarize_results(tmp_path, 0) == 1
    assert "Total: **FAILED**" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize(
    "xml",
    [
        None,
        "broken",
        "<testsuites/>",
        '<testsuite tests="0"/>',
        '<testsuite tests="1" errors="1"/>',
    ],
)
def test_nightly_summary_rejects_lost_or_invalid_raw_shards(tmp_path, xml):
    (tmp_path / "test.xml").write_text('<testsuite tests="1"/>')
    raw = tmp_path / "raw/ppu-attention-junit"
    raw.mkdir(parents=True)
    (raw / "other.xml").write_text('<testsuite tests="1"/>')
    (raw / "step-shard0.log").write_text("shard output")
    if xml is not None:
        (raw / "step-shard0.xml").write_text(xml)
    assert nightly.summarize_results(tmp_path, 0) == 1
    assert "Total: **FAILED**" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize(
    "release,label",
    [
        ("v0.26.0", "spec_decode_ngram_suffix"),
        ("v0.27.0", "spec_decode_ngram_suffix"),
        ("v0.28.0", "spec_decode_draft_model"),
        ("v0.28.0", "test_qwen3_with_multi_loras"),
    ],
)
def test_nightly_known_empty_step_is_explicitly_skipped(tmp_path, release, label):
    path = tmp_path / "test.xml"
    path.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    assert nightly.record_empty_skip(path, release, label) == 0
    assert ET.parse(path).find("testcase/skipped") is not None
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw/step.xml").write_text(path.read_text())
    assert nightly.summarize_results(tmp_path, 0) == 0
    assert "Total: **SKIPPED**" in (tmp_path / "summary.md").read_text()


def test_nightly_empty_allowlist_does_not_hide_failures_or_other_steps(tmp_path):
    path = tmp_path / "test.xml"
    path.write_text('<testsuite tests="0"/>')
    assert nightly.record_empty_skip(path, "v0.28.0", "spec_decode_eagle") == 1
    path.write_text('<testsuite tests="1" errors="1"/>')
    assert nightly.record_empty_skip(path, "v0.28.0", "spec_decode_draft_model") == 1


@pytest.mark.parametrize(
    "area,unsupported",
    [
        ("cuda", "multi"),
        ("entrypoints_llm", "multi"),
        ("distributed", "single"),
        ("models_distributed", "single"),
        ("weight_loading", "single"),
        ("compile", "single"),
    ],
)
@pytest.mark.parametrize("mode", ["", "single", "multi", "all", "invalid"])
def test_nightly_runner_rejects_unsupported_modes_before_setup(area, unsupported, mode):
    result = subprocess.run(
        [
            "bash",
            "-c",
            'mkdir() { exit 97; }; source "$1"',
            "bash",
            str(ROOT / f"scripts/ppu/test-area-ppu-{area}.sh"),
        ],
        env={**os.environ, "TEST_MODE": mode, "CUDA_VISIBLE_DEVICES": "6,7"},
        text=True,
        capture_output=True,
        timeout=5,
    )
    expected = 2 if mode in (unsupported, "invalid") else 97
    assert result.returncode == expected, result.stdout + result.stderr
    if expected == 2:
        assert f"invalid TEST_MODE '{mode}'" in result.stderr


@pytest.mark.parametrize(
    "expected,visible", [(1, 0), (1, 1), (2, 1), (2, 2), (2, 4), (0, 2)]
)
def test_nightly_requires_allocated_devices_before_install(
    monkeypatch, expected, visible
):
    text = (ROOT / "scripts/ppu/ppu_ci_nightly_install_dependency.sh").read_text()
    marker = "python3 - \"${PPU_EXPECTED_DEVICES}\" <<'PY'\n"
    assert text.index(marker) < text.index("${PIP_INSTALL} --force-reinstall")
    code = text.split(marker, 1)[1].split("\nPY\n", 1)[0]
    monkeypatch.setattr(sys, "argv", ["-", str(expected)])
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: visible)),
    )
    if expected < 1 or visible < expected:
        with pytest.raises(
            SystemExit, match=f"expected at least {expected}.*got {visible}"
        ):
            exec(code, {})
    else:
        exec(code, {})


@pytest.mark.parametrize("upgrade", ["", "--upgrade", "-U"])
def test_nightly_upgrade_does_not_short_circuit_on_offline_toolchain(tmp_path, upgrade):
    text = (ROOT / "scripts/ppu/ppu_ci_nightly_install_dependency.sh").read_text()
    function = (
        "_pip_retry() {" + text.split("_pip_retry() {", 1)[1].split("\n}", 1)[0] + "\n}"
    )
    (tmp_path / "toolchain.whl").touch()
    result = subprocess.run(
        ["bash"],
        input=f"""
set -euo pipefail
PIP_INSTALL=mock_pip
PPU_TOOLCHAIN_DIR="$TEST_TMP"
PPU_WHEELHOUSE="$TEST_TMP"
PPU_PIP_INDEX=https://example.invalid/simple
mock_pip() {{ printf '%s\\n' "$@"; }}
{function}
_pip_retry {upgrade} scipy scikit-learn
""",
        text=True,
        capture_output=True,
        env={**os.environ, "TEST_TMP": str(tmp_path)},
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("--no-index" in result.stdout) == (not upgrade)
    assert ("https://example.invalid/simple" in result.stdout) == bool(upgrade)


@pytest.mark.parametrize("assigned", ["3", "6,7", "8,10,12,14", "", None])
def test_nightly_payload_binds_assigned_devices_before_setup(assigned):
    steps = workflow("ppu-build-nightly.yml")["jobs"]["test"]["steps"]
    command = next(s["run"] for s in steps if s.get("id") == "payload")
    prefix = command.split("<<'RUNEOF'\n", 1)[1].split("\ncd /workspace/source", 1)[0]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    env.pop("NVIDIA_VISIBLE_DEVICES", None)
    if assigned is not None:
        env["NVIDIA_VISIBLE_DEVICES"] = assigned
    result = subprocess.run(
        ["bash"],
        input=prefix + '\nprintf "%s" "$CUDA_VISIBLE_DEVICES"\n',
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )
    if assigned:
        assert result.returncode == 0, result.stderr
        assert result.stdout == assigned
    else:
        assert result.returncode != 0
        assert "Assigned devices are unavailable" in result.stderr
        assert not result.stdout


@pytest.mark.parametrize("area", nightly.AREAS, ids=lambda area: area.name)
def test_area_steps_use_assigned_devices_without_renumbering(tmp_path, area):
    text = (nightly.SCRIPT_DIR / area.script).read_text()
    lines = text.splitlines()
    devices = next(line for line in lines if "read -ra PPU_DEVICES" in line)
    function = (
        "_run_step() {" + text.split("_run_step() {", 1)[1].split("\n}", 1)[0] + "\n}"
    )
    singles = [
        line.split(" _run_step", 1)[0] + " _run_step single 1 tests/mock.py"
        for line in lines
        if line.strip().startswith("CUDA_VISIBLE_DEVICES=") and " _run_step" in line
    ]
    result = subprocess.run(
        ["bash"],
        input=f"""
set -euo pipefail
TOTAL_RC=0
STEP_LABELS_LIST=""
TMP_JUNIT="$TEST_TMP"
PYTEST_EXTRA=()
{devices}
pytest() {{ printf 'devices=%s\\n' "$CUDA_VISIBLE_DEVICES"; }}
{function}
{chr(10).join(singles)}
_run_step multi 1 tests/mock.py
_run_step sharded 4 tests/mock.py
exit "$TOTAL_RC"
""",
        env={
            **os.environ,
            "TEST_TMP": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": "8,10,12,14",
            "PPU_RELEASE": "v0.28.0",
        },
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = [
        line for line in result.stdout.splitlines() if line.startswith("devices=")
    ]
    assert observed[: len(singles) + 1] == ["devices=8"] * len(singles) + [
        "devices=8,10,12,14"
    ]
    for shard, device in enumerate(["8", "10", "12", "14"]):
        assert (
            tmp_path / f"sharded-shard{shard}.log"
        ).read_text() == f"devices={device}\n"


@pytest.mark.parametrize("area", ["attention", "samplers", "benchmarks"])
@pytest.mark.parametrize("visible", [None, "6,7"])
def test_pr_shared_area_setup_keeps_legacy_devices_without_nightly_binding(
    area, visible
):
    env = dict(os.environ)
    for name in ("PPU_RELEASE", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        env.pop(name, None)
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    result = subprocess.run(
        [
            "bash",
            "-c",
            'mkdir() { printf "%s" "${PPU_DEVICES[*]}"; exit 97; }; source "$1"',
            "bash",
            str(ROOT / f"scripts/ppu/test-area-ppu-{area}.sh"),
        ],
        env={**env, "TEST_MODE": "single"},
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 97, result.stdout + result.stderr
    assert result.stdout == "0"


@pytest.mark.parametrize("area", ["lora", "spec_decode"])
@pytest.mark.parametrize("shards", [1, 2])
def test_nightly_runner_rejects_unexpected_empty_steps(tmp_path, area, shards):
    text = (ROOT / f"scripts/ppu/test-area-ppu-{area}.sh").read_text()
    function = (
        "_run_step() {" + text.split("_run_step() {", 1)[1].split("\n}", 1)[0] + "\n}"
    )
    script = f'''
set -euo pipefail
TOTAL_RC=0
STEP_LABELS_LIST=""
PPU_RELEASE=v0.28.0
TMP_JUNIT="$TEST_TMP"
PYTEST_EXTRA=()
PPU_DEVICES=(6 7)
pytest() {{ return 5; }}
python3() {{ "{sys.executable}" "$@"; }}
{function}
_run_step unexpected_step {shards} tests/mock.py
exit "$TOTAL_RC"
'''
    result = subprocess.run(
        ["bash"],
        input=script,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env={**os.environ, "TEST_TMP": str(tmp_path)},
    )
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize(
    "area_rc,install_rc,copy_failure",
    [(0, 0, False), (7, 0, False), (0, 9, False), (0, 0, True)],
)
def test_nightly_payload_recovers_results_and_propagates_failures(
    tmp_path, area_rc, install_rc, copy_failure
):
    job = workflow("ppu-build-nightly.yml")["jobs"]["test"]
    step = next(s for s in job["steps"] if s.get("id") == "payload")
    payload = step["run"].split("<<'RUNEOF'\n", 1)[1].split("\nRUNEOF", 1)[0]
    payload = payload.replace("cd /workspace/source/source", 'cd "$TEST_TMP"')
    payload = payload.replace("/tmp/ppu-*-junit", '"$TEST_TMP"/ppu-*-junit')
    payload = payload.replace('"$(mktemp -d)/hub"', '"$TEST_TMP/private/hub"')
    mocks = f'''
python3() {{ "{sys.executable}" "{ROOT}/$1" "${{@:2}}"; }}
bash() {{
  [[ "$CUDA_VISIBLE_DEVICES" == "6,7" ]] || return 96
  if [[ "$1" == *install_dependency.sh ]]; then
    echo installer-output
    return {install_rc}
  fi
  printf '<testsuite name="unit" tests="1"/>' > test-results/test.xml
  mkdir -p ppu-attention-junit
  cp test-results/test.xml ppu-attention-junit/step-shard0.xml
  echo shard-output > ppu-attention-junit/step-shard0.log
  echo area-output
  return {area_rc}
}}
cp() {{
  if [[ "$2" == "test-results/." && {int(copy_failure)} == 1 ]]; then return 1; fi
  command cp "$@"
}}
'''
    nas = tmp_path / "nas"
    result = subprocess.run(
        ["bash"],
        input=mocks + payload,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "TEST_TMP": str(tmp_path),
            "NVIDIA_VISIBLE_DEVICES": "6,7",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PPU_RELEASE": "v0.28.0",
            "AREA": "attention",
            "AREA_SCRIPT": "test-area-ppu-attention.sh",
            "PPU_SOURCE_SHA": "a" * 40,
            "PPU_SUMMARY_NAS_DIR": str(nas),
            "HF_HUB_CACHE": str(tmp_path / "shared"),
        },
    )
    assert bool(result.returncode) == bool(area_rc or install_rc or copy_failure), (
        result.stdout + result.stderr
    )
    if copy_failure:
        assert not (nas / "summary.md").exists()
        return
    assert "installer-output" in (nas / "run.log").read_text()
    if install_rc:
        assert "area-output" not in (nas / "run.log").read_text()
        assert not (nas / "test.xml").exists()
    else:
        assert "area-output" in (nas / "run.log").read_text()
        raw = nas / "raw/ppu-attention-junit"
        assert (raw / "step-shard0.xml").read_text() == (nas / "test.xml").read_text()
        assert (raw / "step-shard0.log").read_text() == "shard-output\n"
    expected = "FAILED" if area_rc or install_rc else "PASSED"
    assert f"Total: **{expected}**" in (nas / "summary.md").read_text()
    assert int((nas / "exit-code.txt").read_text()) == int(bool(area_rc or install_rc))


def test_nightly_smoke_and_artifacts_are_device_scoped():
    config = workflow("ppu-nightly-ci.yml")
    dispatch_inputs = config[True]["workflow_dispatch"]["inputs"]
    assert dispatch_inputs["zw_smoke"]["default"] is False
    assert dispatch_inputs["device"]["type"] == "choice"
    assert dispatch_inputs["device"]["default"] == "OAM-810E"
    assert dispatch_inputs["device"]["options"] == ["OAM-810E", "ZW-M890P"]
    assert config["concurrency"] == {
        "group": "nightly-ppu-releases",
        "cancel-in-progress": False,
    }
    for release in nightly.SUPPORTED_RELEASES:
        job = config["jobs"]["release-" + release.replace(".", "-")]
        assert job["secrets"] == {
            "PPU_ARTIFACTORY_USER": "${{ secrets.PPU_ARTIFACTORY_USER }}",
            "PPU_ARTIFACTORY_PASSWORD": "${{ secrets.PPU_ARTIFACTORY_PASSWORD }}",
        }
        inputs = job["with"]
        assert (
            inputs["device"]
            == "${{ inputs.zw_smoke && 'ZW-M890P' || inputs.device || 'OAM-810E' }}"
        )
        assert (
            inputs["areas"] == "${{ inputs.zw_smoke && 'attention' || inputs.areas }}"
        )
    release_config = workflow("ppu-build-nightly.yml")
    assert release_config[True]["workflow_call"]["secrets"] == {
        "PPU_ARTIFACTORY_USER": {"required": True},
        "PPU_ARTIFACTORY_PASSWORD": {"required": True},
    }
    jobs = release_config["jobs"]
    action = next(s["with"] for s in jobs["test"]["steps"] if s.get("id") == "ppu")
    expected_devices = f"PPU_EXPECTED_DEVICES={action['nproc_per_node']}"
    assert expected_devices in action["extra_env"].split(",")
    assert jobs["build-wheel"]["needs"] == "select-areas"
    upload = next(
        s for s in jobs["test"]["steps"] if s.get("name") == "Upload test results"
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    for key in ("inputs.release", "inputs.device", "matrix.area", "github.run_attempt"):
        assert key in upload["with"]["name"]
