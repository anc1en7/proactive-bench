"""Tests for Workspace isolation and execution (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env.base import MissingParameterError, UnknownAppError, UnknownToolError
from proactive_bench.env.parser import Command, parse_command
from proactive_bench.env.workspace import Workspace


def test_default_workspace_has_all_four_apps() -> None:
    ws = Workspace()
    assert set(ws.snapshot()) == {"repo", "chat", "pager", "cal"}
    assert ws.snapshot()["repo"] == {"prs": [], "issues": [], "commits": []}


def test_unknown_app_key_rejected() -> None:
    with pytest.raises(UnknownAppError):
        Workspace({"slack": {}})


def test_constructor_deep_copies_caller_state() -> None:
    original = {"prs": [{"number": 1}], "issues": [], "commits": []}
    ws = Workspace({"repo": original})
    original["prs"].append({"number": 2})
    assert len(ws.state("repo")["prs"]) == 1


def test_clone_is_isolated_from_parent() -> None:
    ws = Workspace()
    clone = ws.clone()
    clone.state("repo").setdefault("prs", []).append({"number": 1})
    assert ws.state("repo")["prs"] == []
    ws.state("chat").setdefault("channels", []).append({"name": "eng", "messages": []})
    assert clone.state("chat")["channels"] == []


def test_execute_read_and_write_with_result_isolation() -> None:
    ws = Workspace(
        {"chat": {"channels": [{"name": "eng", "messages": []}]}}
    )
    write = ws.execute(
        parse_command(
            {"app": "chat", "function": "send_message",
             "parameter": {"channel": "eng", "text": "hello"}}
        )
    )
    assert write["ok"] is True
    message = write["message"]
    message["text"] = "mutated-by-caller"
    assert ws.state("chat")["channels"][0]["messages"][0]["text"] == "hello"
    read = ws.execute(parse_command({"app": "chat", "function": "list_channels"}))
    assert read["channels"] == [{"name": "eng", "message_count": 1}]


def test_execute_validates_tool_and_params() -> None:
    ws = Workspace()
    with pytest.raises(UnknownToolError):
        ws.execute(Command(app="repo", function="nope", parameter={}))
    with pytest.raises(MissingParameterError):
        ws.execute(Command(app="repo", function="get_pr", parameter={}))


def test_snapshot_is_deep_copy() -> None:
    ws = Workspace()
    snap = ws.snapshot()
    snap["repo"]["prs"].append("x")
    assert ws.state("repo")["prs"] == []
