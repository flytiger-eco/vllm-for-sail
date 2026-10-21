# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""CPU-only contracts for the on-demand PPU CI workflows."""

from pathlib import Path

import yaml

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
        "pull-requests": "read",
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
        assert job["secrets"] == "inherit"


def test_build_checks_out_exact_head_and_names_retry_artifact():
    config = workflow("ppu-build-pr.yml")
    assert set(config[True]) == {"workflow_call"}
    assert set(config[True]["workflow_call"]["inputs"]) == {
        "pr_head_sha",
        "base_ref",
    }
    job = config["jobs"]["build-wheel"]
    assert job["name"] == "PPU wheel build"
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
        if step.get("uses") == "actions/upload-artifact@v4"
    )
    assert upload["with"]["name"] == "${{ steps.version.outputs.artifact_name }}"
    assert upload["with"]["overwrite"] is True
    assert '"${{ inputs.base_ref }}"' in text


def test_area_jobs_have_read_only_tokens_and_exact_sha_checkout():
    for area in ("attention", "samplers", "benchmarks"):
        config = workflow(f"ppu-{area}-pr.yml")
        assert config["permissions"] == {"contents": "read", "actions": "read"}
        job = next(iter(config["jobs"].values()))
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


def test_lightweight_checks_replace_non_ppu_legacy_jobs():
    config = workflow("pr-light-checks.yml")
    assert set(config[True]) == {"pull_request_target", "pull_request_review"}
    jobs = config["jobs"]
    assert set(jobs) == {"secret-scan", "smoke-test", "ai-code-review"}
    for job_id in ("secret-scan", "smoke-test"):
        checkout = checkouts(jobs[job_id])[0]
        assert checkout["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
        assert checkout["with"]["persist-credentials"] is False
        assert jobs[job_id]["permissions"] == {"contents": "read"}
    trufflehog = next(
        step["uses"]
        for step in jobs["secret-scan"]["steps"]
        if step.get("uses", "").startswith("trufflesecurity/trufflehog@")
    )
    assert trufflehog == (
        "trufflesecurity/trufflehog@e7a8d17d2f726cf414310a625b55cca7dc9c898e"
    )
    assert jobs["ai-code-review"]["uses"].endswith(
        "@c03108c83177ebf0038bb5f49b0e15db9b2f78de"
    )


def test_legacy_automatic_pipeline_is_cancelled_before_heavy_jobs():
    config = workflow("cancel-legacy-ppu-ci.yml")
    assert config[True]["workflow_run"] == {
        "workflows": ["CI Pipeline"],
        "types": ["requested", "in_progress"],
    }
    job = config["jobs"]["cancel"]
    assert job["permissions"] if "permissions" in job else config["permissions"]
    assert "cancelWorkflowRun" in str(job)


def test_approval_recorder_no_longer_needs_vllm_self_hosted_runner():
    job = workflow("record-ci-approval.yml")["jobs"]["approved"]
    assert job["runs-on"] == "ubuntu-latest"
    assert "vllm-runners" not in str(job)
