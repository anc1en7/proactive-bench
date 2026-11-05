"""Integration: a scripted multi-app tool trace executes end-to-end (Phase 2 exit)."""

from __future__ import annotations

import pytest

from proactive_bench.env import Workspace, classify_action_level, parse_command
from proactive_bench.env.base import ToolExecutionError


def make_workspace() -> Workspace:
    return Workspace(
        {
            "repo": {
                "prs": [
                    {
                        "number": 1,
                        "title": "fix flaky test",
                        "branch": "fix-flaky",
                        "base": "main",
                        "draft": False,
                        "status": "open",
                        "checks": [{"name": "ci", "status": "fail"}],
                        "reviewers": [],
                        "comments": [],
                        "author": "dev",
                    }
                ],
                "issues": [],
                "commits": [{"sha": "abc123", "message": "wip", "reverted": False}],
            },
            "chat": {"channels": [{"name": "eng", "messages": []}]},
            "pager": {
                "alerts": [
                    {"id": "a1", "severity": "sev2", "service": "api",
                     "status": "firing", "summary": "latency spike"}
                ],
                "runbooks": [],
                "deploy_windows": [],
                "rollback_plans": [],
                "oncall": {"primary": "sam", "paged": False, "page_log": []},
            },
            "cal": {"events": [], "focus_blocks": []},
        }
    )


def test_scripted_tool_trace_executes() -> None:
    ws = make_workspace()
    results: list[dict] = []

    steps = [
        {"app": "repo", "function": "list_prs", "parameter": {}},
        {"app": "repo", "function": "get_checks", "parameter": {"number": 1}},
        {"app": "repo", "function": "retrigger_ci", "parameter": {"number": 1}},
        {"app": "chat", "function": "send_message",
         "parameter": {"channel": "eng", "text": "CI re-running"}},
        {"app": "pager", "function": "ack_alert", "parameter": {"id": "a1"}},
        {"app": "pager", "function": "page_oncall",
         "parameter": {"alert_id": "a1", "reason": "customer impact"}},
        {"app": "cal", "function": "create_event",
         "parameter": {"title": "postmortem", "start": "2026-11-12T16:00:00Z",
                       "end": "2026-11-12T17:00:00Z"}},
        {"app": "repo", "function": "revert_commit", "parameter": {"sha": "abc123"}},
    ]
    for raw in steps:
        results.append(ws.execute(parse_command(raw)))

    assert results[0]["prs"][0]["number"] == 1
    assert results[1]["checks"][0]["status"] == "fail"
    assert results[2]["checks"][0]["status"] == "pending"
    assert results[3]["message"]["text"] == "CI re-running"
    assert results[4]["alert"]["status"] == "acked"
    assert ws.state("pager")["oncall"]["paged"] is True
    assert results[6]["event"]["id"] == "e1"
    assert results[7]["commit"]["reverted"] is True

    # Merge while checks are pending must fail (agent-visible error)...
    with pytest.raises(ToolExecutionError, match="failing or pending"):
        ws.execute(parse_command({"app": "repo", "function": "merge_pr",
                                  "parameter": {"number": 1}}))
    # ...then CI passes and the merge succeeds.
    ws.state("repo")["prs"][0]["checks"][0]["status"] = "pass"
    merged = ws.execute(parse_command({"app": "repo", "function": "merge_pr",
                                       "parameter": {"number": 1}}))
    assert merged["pr"]["status"] == "merged"

    snapshot = ws.snapshot()
    assert snapshot["chat"]["channels"][0]["messages"][0]["text"] == "CI re-running"
    assert snapshot["pager"]["oncall"]["page_log"][0]["alert_id"] == "a1"
    assert snapshot["cal"]["events"][0]["status"] == "confirmed"
    assert snapshot["repo"]["commits"][0]["reverted"] is True


def test_trace_actions_classify_to_expected_levels() -> None:
    assert classify_action_level({"app": "repo", "function": "list_prs"}) == "L0"
    assert classify_action_level(
        {"app": "repo", "function": "retrigger_ci", "parameter": {"number": 1}}
    ) == "L1"
    assert classify_action_level(
        {"app": "chat", "function": "send_message",
         "parameter": {"channel": "eng", "text": "x"}}
    ) == "L1"
    assert classify_action_level(
        {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    ) == "L3"
    assert classify_action_level(None) == "L0"


def test_trace_checkpoint_isolation() -> None:
    ws = make_workspace()
    checkpoint = ws.clone()
    checkpoint.execute(parse_command({"app": "repo", "function": "merge_pr",
                                      "parameter": {"number": 1, "force": True}}))
    checkpoint.execute(parse_command({"app": "chat", "function": "send_message",
                                      "parameter": {"channel": "eng", "text": "merged!"}}))
    assert ws.state("repo")["prs"][0]["status"] == "open"
    assert ws.state("chat")["channels"][0]["messages"] == []
    # Original inputs to Workspace() are also untouched after a full trace.
    assert checkpoint.snapshot()["repo"]["prs"][0]["status"] == "merged"
