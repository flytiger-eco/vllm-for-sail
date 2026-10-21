# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cancel queued or running PPU CI runs that are no longer eligible."""

import json
import os
import urllib.error

from .ppu_ci_command import WORKFLOW, Controller, GitHub, Rejected, parse_title


def run_is_eligible(control, run, pr):
    identity = parse_title(run.get("display_title", ""))
    if not identity or identity[2] != (pr.get("head") or {}).get("sha"):
        return False
    try:
        control.authorize(pr, identity[3])
    except Rejected:
        return False
    return True


def active_executor_runs(control):
    for run in control.pages(
        f"/actions/workflows/{WORKFLOW}/runs",
        "workflow_runs",
        limit=10,
        event="workflow_dispatch",
    ):
        if (
            run.get("status") != "completed"
            and parse_title(run.get("display_title", ""))
            and run.get("path") == f".github/workflows/{WORKFLOW}"
            and run.get("head_branch") == control.default_branch
            and run.get("repository", {}).get("full_name") == control.repo
        ):
            yield run


def run(control):
    event_pr = control.event.get("pull_request")
    if event_pr:
        candidates = [(item, event_pr) for item in control.find_runs(event_pr)]
    else:
        prs = {}
        candidates = []
        for item in active_executor_runs(control):
            number = int(parse_title(item["display_title"])[1])
            if number not in prs:
                try:
                    prs[number] = control.api("GET", f"/pulls/{number}")
                except urllib.error.HTTPError as exc:
                    if exc.code != 404:
                        raise
                    prs[number] = None
            candidates.append((item, prs[number]))
    for item, pr in candidates:
        if item["status"] == "completed":
            continue
        if pr is not None and run_is_eligible(control, item, pr):
            continue
        control.cancel_run(item, "Superseded or no longer authorized")


def main(env=None, github=None):
    env = os.environ if env is None else env
    with open(env["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    github = github or GitHub(
        env["GITHUB_TOKEN"], env.get("GITHUB_API_URL", "https://api.github.com")
    )
    run(Controller(github, event, env))


if __name__ == "__main__":
    main()
