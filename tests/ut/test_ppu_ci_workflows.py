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
