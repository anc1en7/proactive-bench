"""Simulated GitHub-like repo app: PRs, checks, issues, commits."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    as_int,
)


def initial_state() -> dict[str, Any]:
    return {"prs": [], "issues": [], "commits": []}


def _find_pr(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for pr in state.setdefault("prs", []):
        if as_int(pr.get("number", 0), "number") == wanted:
            return pr
    raise ToolExecutionError(f"no PR with number {wanted}")


def _find_issue(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for issue in state.setdefault("issues", []):
        if as_int(issue.get("number", 0), "number") == wanted:
            return issue
    raise ToolExecutionError(f"no issue with number {wanted}")


def _find_commit(state: dict[str, Any], sha: Any) -> dict[str, Any]:
    if not isinstance(sha, str) or not sha:
        raise InvalidCommandError(f"sha must be a non-empty string, got {sha!r}")
    for commit in state.setdefault("commits", []):
        if commit.get("sha") == sha:
            return commit
    raise ToolExecutionError(f"no commit with sha {sha!r}")


def _find_commentable(state: dict[str, Any], number: Any) -> dict[str, Any]:
    wanted = as_int(number, "number")
    for entity in [*state.setdefault("prs", []), *state.setdefault("issues", [])]:
        if as_int(entity.get("number", 0), "number") == wanted:
            return entity
    raise ToolExecutionError(f"no PR or issue with number {wanted}")


# --- reads -----------------------------------------------------------------


def _list_prs(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"prs": list(state.setdefault("prs", []))}


def _get_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"pr": _find_pr(state, params["number"])}


def _get_checks(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"checks": list(_find_pr(state, params["number"]).get("checks", []))}


def _list_issues(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"issues": list(state.setdefault("issues", []))}


def _get_issue(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"issue": _find_issue(state, params["number"])}


# --- writes ----------------------------------------------------------------


def _post_comment(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    body = params["body"]
    if not isinstance(body, str) or not body:
        raise InvalidCommandError(f"body must be a non-empty string, got {body!r}")
    entity = _find_commentable(state, params["number"])
    comment = {"author": "agent", "body": body}
    entity.setdefault("comments", []).append(comment)
    return {"ok": True, "comment": comment}


def _request_review(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    reviewer = params["reviewer"]
    if not isinstance(reviewer, str) or not reviewer:
        raise InvalidCommandError(f"reviewer must be a non-empty string, got {reviewer!r}")
    pr = _find_pr(state, params["number"])
    reviewers = pr.setdefault("reviewers", [])
    if reviewer not in reviewers:
        reviewers.append(reviewer)
    return {"ok": True, "reviewers": list(reviewers)}


def _retrigger_ci(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pr = _find_pr(state, params["number"])
    for check in pr.setdefault("checks", []):
        check["status"] = "pending"
    return {"ok": True, "checks": list(pr.get("checks", []))}


def _open_draft_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = params["title"]
    branch = params["branch"]
    for key, value in (("title", title), ("branch", branch)):
        if not isinstance(value, str) or not value:
            raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    prs = state.setdefault("prs", [])
    number = max((as_int(pr.get("number", 0), "number") for pr in prs), default=0) + 1
    pr = {
        "number": number,
        "title": title,
        "branch": branch,
        "base": "main",
        "draft": True,
        "status": "open",
        "checks": [],
        "reviewers": [],
        "comments": [],
        "author": "agent",
    }
    prs.append(pr)
    return {"ok": True, "pr": pr}


def _merge_pr(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    pr = _find_pr(state, params["number"])
    force = params.get("force", False)
    if not isinstance(force, bool):
        raise InvalidCommandError(f"force must be a boolean, got {force!r}")
    if pr.get("draft"):
        raise ToolExecutionError(f"cannot merge PR #{pr['number']}: it is a draft")
    not_passing = [c for c in pr.get("checks", []) if c.get("status") != "pass"]
    if not_passing and not force:
        names = [c.get("name", "?") for c in not_passing]
        raise ToolExecutionError(
            f"cannot merge PR #{pr['number']}: failing or pending checks {names} "
            "(force=true to override)"
        )
    pr["status"] = "merged"
    pr["draft"] = False
    return {"ok": True, "pr": pr}


def _revert_commit(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    commit = _find_commit(state, params["sha"])
    if commit.get("reverted"):
        raise ToolExecutionError(f"commit {commit['sha']} already reverted")
    commit["reverted"] = True
    return {"ok": True, "commit": commit}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="repo", name="list_prs", kind="read", handler=_list_prs),
    ToolSpec(app="repo", name="get_pr", kind="read", handler=_get_pr, required=("number",)),
    ToolSpec(
        app="repo", name="get_checks", kind="read", handler=_get_checks, required=("number",)
    ),
    ToolSpec(app="repo", name="list_issues", kind="read", handler=_list_issues),
    ToolSpec(app="repo", name="get_issue", kind="read", handler=_get_issue, required=("number",)),
    ToolSpec(
        app="repo",
        name="post_comment",
        kind="write",
        handler=_post_comment,
        required=("number", "body"),
    ),
    ToolSpec(
        app="repo",
        name="request_review",
        kind="write",
        handler=_request_review,
        required=("number", "reviewer"),
    ),
    ToolSpec(
        app="repo", name="retrigger_ci", kind="write", handler=_retrigger_ci, required=("number",)
    ),
    ToolSpec(
        app="repo",
        name="open_draft_pr",
        kind="write",
        handler=_open_draft_pr,
        required=("title", "branch"),
    ),
    ToolSpec(app="repo", name="merge_pr", kind="write", handler=_merge_pr, required=("number",)),
    ToolSpec(
        app="repo", name="revert_commit", kind="write", handler=_revert_commit, required=("sha",)
    ),
)
