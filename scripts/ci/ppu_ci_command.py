# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stdlib-only PPU CI control plane; executor runs are the sole run database."""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import cached_property

COMMANDS = ("/ci run", "/ci run --allow-stale", "/ci retry", "/ci cancel")
HELP = "Commands: " + ", ".join(f"`{command}`" for command in COMMANDS)
READY = {"ready", "ready-run-all-tests"}
WRITE = {"write", "maintain", "admin"}
SUPPORTED_BASES = {"v0.26.0", "v0.27.0", "v0.28.0"}
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
WORKFLOW = "ppu-pr-ci.yml"
ASCII_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def positive_integer(value):
    return (
        bool(value) and value[0] != "0" and all(char in "0123456789" for char in value)
    )


def valid_sha(value):
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def valid_login(value):
    return (
        1 <= len(value) <= 39
        and value[0] in ASCII_ALNUM
        and value[-1] in ASCII_ALNUM
        and all(char in ASCII_ALNUM or char == "-" for char in value)
    )


def parse_title(title):
    parts = title.split()
    if (
        len(parts) != 11
        or parts[:2] != ["PPU", "PR"]
        or not parts[2].startswith("#")
        or parts[3] != "base"
        or parts[5] != "@"
        or parts[7] != "by"
        or parts[9] != "(comment"
        or not parts[10].endswith(")")
    ):
        return None
    number, base_ref, sha = parts[2][1:], parts[4], parts[6]
    login, comment = parts[8], parts[10][:-1]
    if not (
        positive_integer(number)
        and base_ref in SUPPORTED_BASES
        and valid_sha(sha)
        and valid_login(login)
        and positive_integer(comment)
    ):
        return None
    return title, number, sha, login, comment, base_ref


def bot_owned(item):
    user = item.get("user") or {}
    return user.get("type") == "Bot" and user.get("login") == "github-actions[bot]"


def marker(comment_id):
    return f"<!-- ppu-ci-command:{comment_id} -->"


def best_effort(operation):
    try:
        return operation()
    except Exception as exc:
        print(f"Best-effort GitHub operation failed: {exc}", file=sys.stderr)


class Rejected(ValueError):
    """An expected refusal that should be explained to the requester."""


def parse_command(event):
    comment = event.get("comment", {})
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or comment.get("user", {}).get("type") != "User"
    ):
        return None
    body = comment.get("body", "").strip()
    if not body or body.split()[0] != "/ci":
        return None
    if body not in COMMANDS:
        raise Rejected(HELP)
    return body


class GitHub:
    """REST/GraphQL JSON transport. HTTP failures retain their status code."""

    def __init__(self, token, api_url="https://api.github.com"):
        self.token, self.api_url = token, api_url.rstrip("/")

    def request(self, method, path, data=None, params=None):
        base = self.api_url.removesuffix("/v3") if path == "/graphql" else self.api_url
        url = base + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode() if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
        return json.loads(body) if body else None

    def graphql(self, query, variables):
        result = self.request(
            "POST", "/graphql", {"query": query, "variables": variables}
        )
        if result.get("errors"):
            raise RuntimeError("GitHub GraphQL request failed")
        return result["data"]


class Controller:
    """Inject request/GraphQL client, event, environment and sleep for CPU tests."""

    def __init__(self, github, event, env=None, sleep=time.sleep):
        self.gh, self.event = github, event
        self.env = os.environ if env is None else env
        self.sleep = sleep
        self.repo = self.env["GITHUB_REPOSITORY"]
        self.root = f"/repos/{self.repo}"
        server = self.env.get("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
        self.url = f"{server}/{self.repo}"
        self.trusted = set(
            self.env.get("CI_TRUSTED_USERS", "").lower().replace(",", " ").split()
        )

    def api(self, method, path, data=None, params=None):
        return self.gh.request(method, self.root + path, data, params)

    def pages(self, path, key=None, limit=None, **params):
        page = 1
        while limit is None or page <= limit:
            result = self.api(
                "GET", path, params={**params, "per_page": 100, "page": page}
            )
            items = result[key] if key else result
            yield from items
            if len(items) < 100:
                break
            page += 1

    def can_write(self, login):
        try:
            result = self.api(
                "GET", f"/collaborators/{urllib.parse.quote(login, safe='')}/permission"
            )
            return result.get("permission") in WRITE
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            return False

    def authorize(self, pr, login):
        if pr["state"] != "open" or pr["base"]["repo"]["full_name"] != self.repo:
            raise Rejected("PR must be open and target this repository.")
        if pr["base"]["ref"] not in SUPPORTED_BASES:
            raise Rejected("PPU CI supports v0.26.0, v0.27.0, and v0.28.0 only.")
        head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name")
        if not head_repo or head_repo != self.repo:
            raise Rejected("PPU CI does not execute code from fork pull requests.")
        writer = self.can_write(login)
        if writer or login.lower() in self.trusted:
            return writer
        if (
            login.lower() != pr["user"]["login"].lower()
            or pr.get("draft")
            or pr.get("author_association", "") not in TRUSTED_ASSOCIATIONS
        ):
            raise Rejected(
                "Only maintainers, trusted users, or an authorized non-draft "
                "PR author may run CI."
            )
        if READY.intersection(label["name"] for label in pr.get("labels", [])):
            return False
        owner, name = self.repo.split("/")
        result = self.gh.graphql(
            "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name){pullRequest(number:$number){reviewDecision}}}",
            {"owner": owner, "name": name, "number": pr["number"]},
        )
        decision = ((result.get("repository") or {}).get("pullRequest") or {}).get(
            "reviewDecision"
        )
        if decision == "APPROVED":
            latest = {}
            for review in sorted(
                self.pages(f"/pulls/{pr['number']}/reviews"),
                key=lambda r: (r.get("submitted_at") or "", r["id"]),
            ):
                if review.get("user") and review["state"] in {
                    "APPROVED",
                    "CHANGES_REQUESTED",
                    "DISMISSED",
                }:
                    latest[review["user"]["login"].lower()] = review["state"]
            for reviewer, state in latest.items():
                if state == "APPROVED" and (
                    reviewer in self.trusted or self.can_write(reviewer)
                ):
                    return False
        raise Rejected(
            "PR author needs ready/ready-run-all-tests or approval from a "
            "trusted reviewer."
        )

    def authorize_cancel(self, pr, login):
        if (
            self.can_write(login)
            or login.lower() in self.trusted
            or login.lower() == (pr.get("user") or {}).get("login", "").lower()
        ):
            return
        raise Rejected(
            "Only the PR author, maintainers, or trusted users may cancel CI."
        )

    @cached_property
    def default_branch(self):
        return (
            self.event.get("repository", {}).get("default_branch")
            or self.api("GET", "")["default_branch"]
        )

    @property
    def execution_url(self):
        run_id = self.env["GITHUB_RUN_ID"]
        attempt = self.env.get("GITHUB_RUN_ATTEMPT", "1")
        return f"{self.url}/actions/runs/{run_id}/attempts/{attempt}"

    def find_runs(self, pr, head_sha=None, match_base=True):
        found = []
        for run in self.pages(
            f"/actions/workflows/{WORKFLOW}/runs",
            "workflow_runs",
            limit=10,
            event="workflow_dispatch",
            created=">=" + pr["created_at"][:10],
        ):
            title = parse_title(run.get("display_title", ""))
            if (
                title
                and int(title[1]) == pr["number"]
                and (head_sha is None or title[2] == head_sha)
                and (not match_base or title[5] == pr["base"]["ref"])
                and run.get("path") == f".github/workflows/{WORKFLOW}"
                and run.get("head_branch") == self.default_branch
                and run.get("repository", {}).get("full_name") == self.repo
            ):
                found.append(run)
        return sorted(found, key=lambda run: run["id"], reverse=True)

    def find_comment(self, number, tag):
        for comment in self.pages(f"/issues/{number}/comments"):
            if bot_owned(comment) and tag in (comment.get("body") or ""):
                return comment

    def reply(self, number, comment_id, text):
        body = f"{marker(comment_id)}\n{text}"
        self.api("POST", f"/issues/{number}/comments", {"body": body})

    def react(self, comment_id, content):
        path = f"/issues/comments/{comment_id}/reactions"
        best_effort(lambda: self.api("POST", path, {"content": content}))

    def status(self, sha, base_ref, state, description):
        payload = dict(
            state=state,
            context=f"PPU CI / {base_ref}",
            description=description[:140],
            target_url=self.execution_url,
        )
        self.api("POST", f"/statuses/{sha}", payload)

    def launch(self, sha, base_ref, description, path, data=None, fallback=None):
        self.status(sha, base_ref, "pending", description)
        try:
            try:
                self.api("POST", path, data)
            except urllib.error.HTTPError as exc:
                if not fallback or not 400 <= exc.code < 500:
                    raise
                self.api("POST", fallback)
        except Exception as exc:
            self.status(sha, base_ref, "error", "CI request failed")
            if isinstance(exc, urllib.error.HTTPError):
                raise Rejected(f"CI request failed (HTTP {exc.code}).") from exc
            raise

    def command(self):
        try:
            action = parse_command(self.event)
        except Rejected:
            action = "help"
        if action is None:
            return
        number, comment = self.event["issue"]["number"], self.event["comment"]
        cid, login = comment["id"], comment["user"]["login"]
        try:
            path = f"/issues/comments/{cid}/reactions"
            reactions = best_effort(lambda: list(self.pages(path))) or []
            if any(
                bot_owned(r) and r["content"] in {"rocket", "-1"} for r in reactions
            ) or self.find_comment(number, marker(cid)):
                return
            self.react(cid, "eyes")
            reaction = "rocket"
            try:
                if action == "help":
                    raise Rejected(HELP)
                pr = self.api("GET", f"/pulls/{number}")
                if action == "/ci cancel":
                    self.authorize_cancel(pr, login)
                    text = self.cancel(pr, login)
                else:
                    self.authorize(pr, login)
                    if action == "/ci retry":
                        text = self.retry(pr, login)
                    else:
                        text = self.run(
                            pr, login, cid, action.endswith("--allow-stale")
                        )
            except Rejected as exc:
                text, reaction = str(exc), "-1"
            self.reply(number, cid, text)
            self.react(cid, reaction)
        except Exception:
            self.react(cid, "confused")
            raise

    def run(self, pr, login, cid, allow_stale):
        sha = pr["head"]["sha"]
        runs = self.find_runs(pr, sha)
        active = next((r for r in runs if r["status"] != "completed"), None)
        if active:
            return f"PPU CI already running: {active['html_url']}"
        completed = next((r for r in runs if r["status"] == "completed"), None)
        base = urllib.parse.quote(pr["base"]["ref"], safe="")
        behind = self.api("GET", f"/compare/{base}...{sha}")["behind_by"]
        if behind and not allow_stale:
            raise Rejected(
                f"PR is {behind} commits behind base. Rebase or use "
                "`/ci run --allow-stale`."
            )
        fresh = self.api("GET", f"/pulls/{pr['number']}")
        if fresh["head"]["sha"] != sha:
            raise Rejected("PR head changed; use /ci run again.")
        if (fresh.get("base") or {}).get("ref") != pr["base"]["ref"]:
            raise Rejected("PR base changed; use /ci run again.")
        writer = self.authorize(fresh, login)
        if completed and not writer:
            raise Rejected(
                f"Already tested: {completed['html_url']}. Use `/ci retry` for failed "
                "jobs or push a new commit; write permission is required to "
                "dispatch again."
            )
        self.launch(
            sha,
            pr["base"]["ref"],
            f"Queued by @{login}",
            f"/actions/workflows/{WORKFLOW}/dispatches",
            {
                "ref": self.default_branch,
                "inputs": {
                    "pr_number": str(pr["number"]),
                    "head_sha": sha,
                    "base_ref": pr["base"]["ref"],
                    "requester": login,
                    "comment_id": str(cid),
                },
            },
        )
        warning = f"Warning: running {behind} commits behind base. " if behind else ""
        deadline = time.monotonic() + 30
        for _ in range(10):
            if time.monotonic() + 3 > deadline:
                break
            self.sleep(3)
            for run in best_effort(lambda: self.find_runs(pr, sha)) or []:
                if parse_title(run["display_title"])[4] == str(cid):
                    return warning + f"PPU CI queued for {sha[:12]}: {run['html_url']}"
        return (
            warning
            + "Dispatch accepted; run link pending: "
            + f"{self.url}/actions/workflows/{WORKFLOW}"
        )

    def cancel_run(self, run, description):
        try:
            self.api("POST", f"/actions/runs/{run['id']}/cancel")
        except urllib.error.HTTPError as exc:
            if exc.code not in {409, 422}:
                raise
            return False
        identity = parse_title(run["display_title"])
        self.status(identity[2], identity[5], "error", description)
        return True

    def cancel(self, pr, login):
        links = []
        for run in self.find_runs(pr, match_base=False):
            if run["status"] != "completed" and self.cancel_run(
                run, f"Cancelled by @{login}"
            ):
                links.append(run["html_url"])
        return (
            f"Cancellation requested for {len(links)} runs: " + " ".join(links)
            if links
            else "No active PPU CI runs (none to cancel)."
        )

    def retry(self, pr, login):
        runs = self.find_runs(pr, pr["head"]["sha"])
        if not runs:
            raise Rejected("No run for this head; use /ci run.")
        run = runs[0]
        if run["status"] != "completed":
            return f"PPU CI already running: {run['html_url']}"
        if run["conclusion"] == "success":
            return f"Nothing to rerun: {run['html_url']}"
        path = f"/actions/runs/{run['id']}"
        jobs = self.pages(path + "/jobs", "jobs", filter="latest")
        built = any(
            j["name"].rsplit("/", 1)[-1].strip().startswith("PPU wheel build")
            and j["conclusion"] == "success"
            for j in jobs
        )
        if built and not any(
            a["name"] == f"ppu-wheels-{run['id']}" and not a.get("expired", True)
            for a in self.pages(path + "/artifacts", "artifacts")
        ):
            raise Rejected("Wheel artifact missing or expired; use /ci run.")
        identity = parse_title(run["display_title"])
        fresh = self.api("GET", f"/pulls/{pr['number']}")
        if fresh["head"]["sha"] != identity[2] or fresh["base"]["ref"] != identity[5]:
            raise Rejected("PR head or base changed; use /ci run again.")
        self.authorize(fresh, login)
        if identity[3].lower() != login.lower():
            try:
                self.authorize(fresh, identity[3])
            except Rejected as exc:
                raise Rejected(
                    "Original requester is no longer authorized; use /ci run."
                ) from exc
        # Native retry can reuse successful validate; report depends on failed ppu.
        self.launch(
            fresh["head"]["sha"],
            fresh["base"]["ref"],
            f"Retry requested by @{login}",
            path + "/rerun-failed-jobs",
            fallback=path + "/rerun",
        )
        attempt = run["run_attempt"] + 1
        return f"PPU CI retry requested: {run['html_url']}/attempts/{attempt}"

    def notify(self):
        pr = self.event.get("pull_request", {})
        if (
            self.event.get("action") != "labeled"
            or self.event.get("label", {}).get("name") not in READY
            or pr.get("state") != "open"
            or pr.get("draft")
            or pr.get("base", {}).get("ref") not in SUPPORTED_BASES
            or (pr.get("head", {}).get("repo") or {}).get("full_name") != self.repo
        ):
            return
        login = pr["user"]["login"]
        tag = "<!-- ppu-ci-authorized -->"
        if (
            login.lower() in self.trusted
            or self.can_write(login)
            or self.find_comment(pr["number"], tag)
        ):
            return
        body = (
            f"{tag}\n@{login}: {HELP}. New commits require `/ci run`. "
            "`--allow-stale` permits running behind the base branch. "
            "Security and lightweight checks remain automatic."
        )
        self.api("POST", f"/issues/{pr['number']}/comments", {"body": body})

    def inputs(self):
        inputs = self.event.get("inputs", {})
        validators = dict(
            pr_number=positive_integer,
            head_sha=valid_sha,
            base_ref=lambda value: value in SUPPORTED_BASES,
            requester=valid_login,
            comment_id=positive_integer,
        )
        for key, validator in validators.items():
            value = inputs.get(key)
            if not isinstance(value, str) or not validator(value):
                raise Rejected(f"Invalid input: {key}")
        return inputs

    def validate(self):
        inputs = self.inputs()
        pr = self.api("GET", f"/pulls/{inputs['pr_number']}")
        if pr["head"]["sha"] != inputs["head_sha"]:
            raise Rejected("PR head changed; use /ci run again.")
        if pr["base"]["ref"] != inputs["base_ref"]:
            raise Rejected("PR base changed; use /ci run again.")
        self.authorize(pr, inputs["requester"])
        output = self.env.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a") as stream:
                stream.write(f"base_ref={pr['base']['ref']}\n")
        self.status(
            inputs["head_sha"],
            inputs["base_ref"],
            "pending",
            f"Running (attempt {self.env.get('GITHUB_RUN_ATTEMPT', '1')})",
        )

    def report(self):
        inputs = self.inputs()
        needs = json.loads(self.env["NEEDS"])
        if not isinstance(needs, dict):
            raise ValueError("NEEDS must be a JSON object")
        stages = (
            ("validate", "ppu")
            if "ppu" in needs
            else ("validate", "build", "attention", "samplers", "benchmarks")
        )
        results = {stage: needs.get(stage, {}).get("result") for stage in stages}
        state, description = "success", "Passed"
        if "cancelled" in results.values():
            state, description = "error", "Cancelled"
        elif any(result != "success" for result in results.values()):
            stage = next(
                stage for stage, result in results.items() if result != "success"
            )
            state, description = "failure", f"{stage} failed"
        description += f" (attempt {self.env.get('GITHUB_RUN_ATTEMPT', '1')})"
        self.status(inputs["head_sha"], inputs["base_ref"], state, description)

        def comment():
            number, cid = inputs["pr_number"], inputs["comment_id"]
            previous = self.find_comment(number, marker(cid))
            text = f"PPU CI: {description}: {self.execution_url}"
            if previous:
                self.api(
                    "PATCH",
                    f"/issues/comments/{previous['id']}",
                    {"body": f"{marker(cid)}\n{text}"},
                )
            else:
                self.reply(number, cid, text)

        best_effort(comment)


def main(argv=None, env=None, github=None, sleep=time.sleep):
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("command", "notify", "validate", "report"))
    mode = parser.parse_args(argv).mode
    try:
        with open(env["GITHUB_EVENT_PATH"]) as stream:
            event = json.load(stream)
        github = github or GitHub(
            env["GITHUB_TOKEN"], env.get("GITHUB_API_URL", "https://api.github.com")
        )
        getattr(Controller(github, event, env, sleep), mode)()
    except Exception as exc:
        if mode != "validate":
            raise
        message = f"PPU CI validation failed: {exc}\n"
        print(message, file=sys.stderr)
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a") as stream:
                stream.write(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
