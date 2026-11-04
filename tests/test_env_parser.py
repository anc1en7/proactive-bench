"""Tests for the command parser (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    MissingParameterError,
    ToolKindError,
    UnknownAppError,
    UnknownToolError,
)
from proactive_bench.env.parser import Command, parse_command, require_kind


def test_parse_command_happy_path() -> None:
    command = parse_command(
        {"app": "repo", "function": "get_pr", "parameter": {"number": 1}}
    )
    assert command == Command(app="repo", function="get_pr", parameter={"number": 1})


def test_parse_command_defaults_missing_parameter() -> None:
    command = parse_command({"app": "repo", "function": "list_prs"})
    assert command.parameter == {}


def test_parse_command_ignores_extra_keys() -> None:
    command = parse_command(
        {
            "event_id": "e1",
            "time": "2026-11-10T09:00:00Z",
            "app": "chat",
            "function": "send_message",
            "parameter": {"channel": "eng", "text": "hi"},
        }
    )
    assert command.app == "chat"
    assert command.function == "send_message"
    assert "event_id" not in vars(command)


def test_parse_command_deep_copies_parameter() -> None:
    raw = {"app": "repo", "function": "post_comment", "parameter": {"number": 1, "body": "x"}}
    command = parse_command(raw)
    raw["parameter"]["body"] = "mutated"
    assert command.parameter["body"] == "x"


def test_parse_command_rejects_structural_errors() -> None:
    for bad in (
        "get_pr",
        42,
        None,
        [],
        {"function": "get_pr"},
        {"app": "repo"},
        {"app": 7, "function": "get_pr"},
        {"app": "repo", "function": ""},
        {"app": "repo", "function": "get_pr", "parameter": None},
        {"app": "repo", "function": "get_pr", "parameter": [1]},
    ):
        with pytest.raises(InvalidCommandError):
            parse_command(bad)


def test_parse_command_unknown_app_or_tool() -> None:
    with pytest.raises(UnknownAppError):
        parse_command({"app": "slack", "function": "post", "parameter": {}})
    with pytest.raises(UnknownToolError):
        parse_command({"app": "repo", "function": "delete_pr", "parameter": {}})


def test_parse_command_missing_required_param() -> None:
    with pytest.raises(MissingParameterError, match="number"):
        parse_command({"app": "repo", "function": "get_pr", "parameter": {}})
    with pytest.raises(MissingParameterError, match="channel, text"):
        parse_command({"app": "chat", "function": "send_message", "parameter": {}})


def test_require_kind_ok_and_mismatch() -> None:
    read_cmd = parse_command({"app": "repo", "function": "list_prs"})
    spec = require_kind(read_cmd, "read")
    assert spec.name == "list_prs"
    with pytest.raises(ToolKindError, match="read tool, not write"):
        require_kind(read_cmd, "write")
    write_cmd = parse_command(
        {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    )
    assert require_kind(write_cmd, "write").kind == "write"
