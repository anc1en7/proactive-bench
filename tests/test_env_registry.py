"""Tests for the tool registry (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import registry
from proactive_bench.env.base import UnknownAppError, UnknownToolError

EXPECTED_TOOLS = {
    "repo": {
        "read": {"list_prs", "get_pr", "get_checks", "list_issues", "get_issue"},
        "write": {"post_comment", "request_review", "retrigger_ci", "open_draft_pr",
                  "merge_pr", "revert_commit"},
    },
    "chat": {
        "read": {"list_channels", "get_thread", "search_messages"},
        "write": {"send_message", "post_standup_update", "pin_message"},
    },
    "pager": {
        "read": {"list_alerts", "get_alert", "get_runbook", "get_deploy_window"},
        "write": {"ack_alert", "page_oncall", "propose_rollback", "execute_rollback"},
    },
    "cal": {
        "read": {"list_events", "get_event", "get_free_slots"},
        "write": {"create_event", "decline_event", "propose_reschedule", "block_focus_time"},
    },
}


def test_apps_match_spec_table_exactly() -> None:
    assert set(registry.APPS) == set(EXPECTED_TOOLS)
    for app, expected in EXPECTED_TOOLS.items():
        specs = registry.APPS[app]
        for kind in ("read", "write"):
            actual = {s.name for s in specs if s.kind == kind}
            assert actual == expected[kind], f"{app} {kind}"


def test_total_tool_count_is_32() -> None:
    assert len(registry.all_tools()) == 32


def test_initial_states_cover_all_apps() -> None:
    assert set(registry.INITIAL_STATES) == set(registry.APPS)
    for factory in registry.INITIAL_STATES.values():
        state = factory()
        assert isinstance(state, dict)
        assert factory() is not state


def test_get_tool_returns_spec() -> None:
    spec = registry.get_tool("repo", "merge_pr")
    assert spec.kind == "write"
    assert spec.required == ("number",)
    assert spec.full_name == "repo.merge_pr"


def test_get_tool_unknown_app_and_tool() -> None:
    with pytest.raises(UnknownAppError, match="slack"):
        registry.get_tool("slack", "post")
    with pytest.raises(UnknownToolError, match="repo.delete_pr"):
        registry.get_tool("repo", "delete_pr")


def test_tool_kind_delegates() -> None:
    assert registry.tool_kind("chat", "send_message") == "write"
    assert registry.tool_kind("cal", "list_events") == "read"


def test_function_names_are_globally_unique() -> None:
    names = [t.name for t in registry.all_tools()]
    assert len(names) == len(set(names))
