"""Tests for the simulated repo app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import repo
from proactive_bench.env.base import InvalidCommandError, ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in repo.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def base_pr(**overrides: object) -> dict:
    pr = {
        "number": 1,
        "title": "fix thing",
        "branch": "fix-1",
        "base": "main",
        "draft": False,
        "status": "open",
        "checks": [{"name": "ci", "status": "pass"}],
        "reviewers": [],
        "comments": [],
        "author": "dev",
    }
    pr.update(overrides)
    return pr


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in repo.TOOLS if s.kind == "read"}
    writes = {s.name for s in repo.TOOLS if s.kind == "write"}
    assert reads == {"list_prs", "get_pr", "get_checks", "list_issues", "get_issue"}
    assert writes == {
        "post_comment",
        "request_review",
        "retrigger_ci",
        "open_draft_pr",
        "merge_pr",
        "revert_commit",
    }


def test_initial_state_is_fresh_each_call() -> None:
    a, b = repo.initial_state(), repo.initial_state()
    assert a == {"prs": [], "issues": [], "commits": []}
    a["prs"].append(1)
    assert b["prs"] == []


def test_list_and_get_pr() -> None:
    state = {"prs": [base_pr()]}
    assert get("list_prs").handler(state, {}) == {"prs": state["prs"]}
    result = get("get_pr").handler(state, {"number": 1})
    assert result["pr"]["title"] == "fix thing"
    assert get("get_pr").handler(state, {"number": "1"})["pr"]["number"] == 1
    with pytest.raises(ToolExecutionError):
        get("get_pr").handler(state, {"number": 99})


def test_get_checks_and_list_issues() -> None:
    state = {
        "prs": [base_pr()],
        "issues": [{"number": 5, "title": "bug", "state": "open", "labels": [], "comments": []}],
    }
    assert get("get_checks").handler(state, {"number": 1})["checks"][0]["status"] == "pass"
    assert get("list_issues").handler(state, {})["issues"][0]["number"] == 5
    assert get("get_issue").handler(state, {"number": 5})["issue"]["title"] == "bug"
    with pytest.raises(ToolExecutionError):
        get("get_issue").handler(state, {"number": 6})


def test_post_comment_appends_to_pr_and_issue() -> None:
    state = {
        "prs": [base_pr()],
        "issues": [{"number": 5, "title": "bug", "state": "open", "labels": [], "comments": []}],
    }
    result = get("post_comment").handler(state, {"number": 1, "body": "lgtm"})
    assert result["ok"] is True
    assert state["prs"][0]["comments"] == [{"author": "agent", "body": "lgtm"}]
    get("post_comment").handler(state, {"number": 5, "body": "repro?"})
    assert len(state["issues"][0]["comments"]) == 1
    with pytest.raises(ToolExecutionError):
        get("post_comment").handler(state, {"number": 404, "body": "x"})


def test_request_review_deduplicates() -> None:
    state = {"prs": [base_pr(reviewers=["alice"])]}
    get("request_review").handler(state, {"number": 1, "reviewer": "alice"})
    get("request_review").handler(state, {"number": 1, "reviewer": "bob"})
    assert state["prs"][0]["reviewers"] == ["alice", "bob"]


def test_retrigger_ci_sets_checks_pending() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    get("retrigger_ci").handler(state, {"number": 1})
    assert state["prs"][0]["checks"][0]["status"] == "pending"


def test_open_draft_pr_assigns_next_number() -> None:
    state = {"prs": [base_pr(number=3)]}
    result = get("open_draft_pr").handler(state, {"title": "new", "branch": "feat-x"})
    pr = result["pr"]
    assert pr["number"] == 4
    assert pr["draft"] is True
    assert pr["status"] == "open"
    assert pr["base"] == "main"
    assert pr["checks"] == [] and pr["reviewers"] == [] and pr["comments"] == []
    assert state["prs"][-1]["title"] == "new"


def test_merge_pr_blocked_by_non_passing_checks() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    with pytest.raises(ToolExecutionError, match="failing or pending"):
        get("merge_pr").handler(state, {"number": 1})
    assert state["prs"][0]["status"] == "open"


def test_merge_pr_blocked_by_pending_checks_unless_forced() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "pending"}])]}
    with pytest.raises(ToolExecutionError):
        get("merge_pr").handler(state, {"number": 1})
    result = get("merge_pr").handler(state, {"number": 1, "force": True})
    assert result["ok"] is True
    assert state["prs"][0]["status"] == "merged"


def test_merge_pr_happy_path_and_draft_block() -> None:
    state = {"prs": [base_pr(checks=[]), base_pr(number=2, draft=True, checks=[])]}
    get("merge_pr").handler(state, {"number": 1})
    assert state["prs"][0]["status"] == "merged"
    with pytest.raises(ToolExecutionError, match="draft"):
        get("merge_pr").handler(state, {"number": 2})


def test_merge_pr_rejects_non_bool_force() -> None:
    state = {"prs": [base_pr(checks=[{"name": "ci", "status": "fail"}])]}
    with pytest.raises(InvalidCommandError, match="force"):
        get("merge_pr").handler(state, {"number": 1, "force": "yes"})


def test_revert_commit_marks_and_blocks_double_revert() -> None:
    state = {"commits": [{"sha": "abc123", "message": "wip", "reverted": False}]}
    get("revert_commit").handler(state, {"sha": "abc123"})
    assert state["commits"][0]["reverted"] is True
    with pytest.raises(ToolExecutionError, match="already reverted"):
        get("revert_commit").handler(state, {"sha": "abc123"})
    with pytest.raises(ToolExecutionError):
        get("revert_commit").handler(state, {"sha": "nope"})


def test_required_params_enforced() -> None:
    from proactive_bench.env.base import MissingParameterError

    with pytest.raises(MissingParameterError, match="missing required parameter"):
        get("get_pr").check_params({})
