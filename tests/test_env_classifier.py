"""Tests for the deterministic action-level classifier (offline)."""

from __future__ import annotations

from typing import get_args

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolError,
    UnknownAppError,
    UnknownToolError,
)
from proactive_bench.env.classifier import ActionLevel, classify_action_level
from proactive_bench.env.parser import Command, parse_command
from proactive_bench.env.registry import all_tools

EXPECTED_LEVELS = {
    # L1 — notify / inform
    "post_comment": "L1",
    "request_review": "L1",
    "retrigger_ci": "L1",
    "send_message": "L1",
    "post_standup_update": "L1",
    "pin_message": "L1",
    "ack_alert": "L1",
    "page_oncall": "L1",
    # L2 — prepare / draft
    "open_draft_pr": "L2",
    "propose_rollback": "L2",
    "propose_reschedule": "L2",
    "block_focus_time": "L2",
    # L3 — execute
    "merge_pr": "L3",
    "revert_commit": "L3",
    "execute_rollback": "L3",
    "create_event": "L3",
    "decline_event": "L3",
}


def test_none_is_l0() -> None:
    assert classify_action_level(None) == "L0"


def test_every_read_tool_is_l0() -> None:
    for spec in all_tools():
        if spec.kind == "read":
            command = Command(app=spec.app, function=spec.name, parameter={})
            assert classify_action_level(command) == "L0", spec.full_name


def test_every_write_tool_has_expected_level() -> None:
    writes = [s for s in all_tools() if s.kind == "write"]
    assert len(writes) == 17
    for spec in writes:
        assert spec.name in EXPECTED_LEVELS, f"missing level for {spec.full_name}"
        command = Command(app=spec.app, function=spec.name, parameter={})
        level = classify_action_level(command)
        assert level == EXPECTED_LEVELS[spec.name], spec.full_name
        assert level in ("L1", "L2", "L3")


def test_command_and_mapping_forms_agree() -> None:
    raw = {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
    command = parse_command(raw)
    assert classify_action_level(command) == classify_action_level(raw) == "L3"


def test_unknown_tool_and_app_raise() -> None:
    with pytest.raises(UnknownToolError):
        classify_action_level({"app": "repo", "function": "delete_pr"})
    with pytest.raises(UnknownAppError):
        classify_action_level({"app": "slack", "function": "post"})


def test_malformed_mapping_raises() -> None:
    for bad in (42, "merge_pr", {"function": "merge_pr"}, {"app": "repo"}):
        with pytest.raises(InvalidCommandError):
            classify_action_level(bad)  # type: ignore[arg-type]


def test_deterministic_across_calls() -> None:
    action = {"app": "pager", "function": "execute_rollback", "parameter": {"plan_id": "rb1"}}
    assert classify_action_level(action) == classify_action_level(action) == "L3"


def test_action_level_type_is_literal() -> None:
    assert set(get_args(ActionLevel)) == {"L0", "L1", "L2", "L3"}


def test_internal_levels_table_is_complete() -> None:
    from proactive_bench.env import classifier as _c

    write_names = {s.name for s in all_tools() if s.kind == "write"}
    assert set(_c._LEVELS) == write_names


def test_write_tool_missing_from_table_raises_tool_error() -> None:
    from proactive_bench.env import classifier as _c

    with pytest.raises(ToolError, match="no action-level mapping"):
        _c._LEVELS.pop("merge_pr")
        try:
            classify_action_level(
                {"app": "repo", "function": "merge_pr", "parameter": {"number": 1}}
            )
        finally:
            _c._LEVELS["merge_pr"] = "L3"
