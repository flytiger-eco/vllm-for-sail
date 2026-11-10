# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""CPU-only contracts for the on-demand PPU CI workflows."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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


@pytest.mark.parametrize("release", nightly.SUPPORTED_RELEASES)
@pytest.mark.parametrize(
    "declaration",
    [
        "WL_ARGS=(tests/selected.py)",
        "TEST_ARGS=(\n tests/selected.py\n)",
        "TEST_ARGS=(\n -m 'distributed(num_gpus=2)'\n tests/selected.py\n)",
        'PYTEST_EXTRA+=(\n --deselect "tests/selected.py::test_case[x]"\n)',
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
    assert nightly.summarize_results(tmp_path, exit_code) == int(expected == "FAILED")
    assert f"Total: **{expected}**" in (tmp_path / "summary.md").read_text()


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
    assert nightly.summarize_results(tmp_path, 0) == 0
    assert "Total: **SKIPPED**" in (tmp_path / "summary.md").read_text()


def test_nightly_empty_allowlist_does_not_hide_failures_or_other_steps(tmp_path):
    path = tmp_path / "test.xml"
    path.write_text('<testsuite tests="0"/>')
    assert nightly.record_empty_skip(path, "v0.28.0", "spec_decode_eagle") == 1
    path.write_text('<testsuite tests="1" errors="1"/>')
    assert nightly.record_empty_skip(path, "v0.28.0", "spec_decode_draft_model") == 1


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
pytest() {{ return 5; }}
python3() {{ "{sys.executable}" "$@"; }}
{function}
_run_step unexpected_step {shards}
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
    assert config[True]["workflow_dispatch"]["inputs"]["zw_smoke"]["default"] is False
    assert config["concurrency"]["queue"] == "max"
    for release in nightly.SUPPORTED_RELEASES:
        inputs = config["jobs"]["release-" + release.replace(".", "-")]["with"]
        assert inputs["device"] == "${{ inputs.zw_smoke && 'ZW-M890P' || 'OAM-810E' }}"
        assert (
            inputs["areas"] == "${{ inputs.zw_smoke && 'attention' || inputs.areas }}"
        )
    jobs = workflow("ppu-build-nightly.yml")["jobs"]
    assert jobs["build-wheel"]["needs"] == "select-areas"
    upload = next(
        s for s in jobs["test"]["steps"] if s.get("name") == "Upload test results"
    )
    assert upload["if"] == "always()"
    assert upload["with"]["if-no-files-found"] == "error"
    for key in ("inputs.release", "inputs.device", "matrix.area", "github.run_attempt"):
        assert key in upload["with"]["name"]
