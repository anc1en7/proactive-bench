"""Tests for the simulated pager/on-call app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import pager
from proactive_bench.env.base import ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in pager.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "alerts": [
            {"id": "a1", "severity": "sev2", "service": "api", "status": "firing",
             "summary": "latency spike"},
            {"id": "a2", "severity": "sev4", "service": "web", "status": "acked",
             "summary": "disk usage"},
        ],
        "runbooks": [{"id": "rb-latency", "title": "Latency", "steps": ["check metrics"]}],
        "deploy_windows": [
            {"id": "dw1", "start": "2026-11-10T18:00:00Z", "end": "2026-11-10T20:00:00Z",
             "frozen": False}
        ],
        "rollback_plans": [
            {"id": "rb1", "service": "api", "reason": "bad deploy", "status": "proposed"}
        ],
        "oncall": {"primary": "sam", "paged": False, "page_log": []},
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in pager.TOOLS if s.kind == "read"}
    writes = {s.name for s in pager.TOOLS if s.kind == "write"}
    assert reads == {"list_alerts", "get_alert", "get_runbook", "get_deploy_window"}
    assert writes == {"ack_alert", "page_oncall", "propose_rollback", "execute_rollback"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = pager.initial_state(), pager.initial_state()
    assert a == {
        "alerts": [],
        "runbooks": [],
        "deploy_windows": [],
        "rollback_plans": [],
        "oncall": {"primary": None, "paged": False, "page_log": []},
    }
    a["alerts"].append(1)
    assert b["alerts"] == []


def test_list_and_get_alerts() -> None:
    state = seed_state()
    assert len(get("list_alerts").handler(state, {})["alerts"]) == 2
    assert get("get_alert").handler(state, {"id": "a1"})["alert"]["severity"] == "sev2"
    with pytest.raises(ToolExecutionError):
        get("get_alert").handler(state, {"id": "a9"})


def test_get_runbook_and_deploy_window() -> None:
    state = seed_state()
    assert get("get_runbook").handler(state, {"id": "rb-latency"})["runbook"]["title"] == "Latency"
    window = get("get_deploy_window").handler(state, {"id": "dw1"})["window"]
    assert window["frozen"] is False
    with pytest.raises(ToolExecutionError):
        get("get_deploy_window").handler(state, {"id": "nope"})


def test_ack_alert_only_from_firing() -> None:
    state = seed_state()
    get("ack_alert").handler(state, {"id": "a1"})
    assert state["alerts"][0]["status"] == "acked"
    with pytest.raises(ToolExecutionError, match="not 'firing'"):
        get("ack_alert").handler(state, {"id": "a1"})
    with pytest.raises(ToolExecutionError, match="not 'firing'"):
        get("ack_alert").handler(state, {"id": "a2"})


def test_page_oncall_logs_and_sets_paged() -> None:
    state = seed_state()
    result = get("page_oncall").handler(state, {"alert_id": "a1", "reason": "customer impact"})
    assert result["ok"] is True
    assert state["oncall"]["paged"] is True
    assert state["oncall"]["page_log"] == [{"alert_id": "a1", "reason": "customer impact"}]
    with pytest.raises(ToolExecutionError):
        get("page_oncall").handler(state, {"alert_id": "a9", "reason": "x"})


def test_propose_and_execute_rollback_lifecycle() -> None:
    state = seed_state()
    result = get("propose_rollback").handler(state, {"service": "web", "reason": "css break"})
    plan = result["plan"]
    assert plan["id"] == "rb2"
    assert plan["status"] == "proposed"
    get("execute_rollback").handler(state, {"plan_id": "rb2"})
    assert state["rollback_plans"][1]["status"] == "executed"
    with pytest.raises(ToolExecutionError, match="not 'proposed'"):
        get("execute_rollback").handler(state, {"plan_id": "rb2"})
    with pytest.raises(ToolExecutionError, match="no rollback plan"):
        get("execute_rollback").handler(state, {"plan_id": "rb1x"})


def test_execute_rollback_requires_existing_proposed_plan() -> None:
    state = seed_state()
    get("execute_rollback").handler(state, {"plan_id": "rb1"})
    assert state["rollback_plans"][0]["status"] == "executed"
    with pytest.raises(ToolExecutionError):
        get("execute_rollback").handler(state, {"plan_id": "rb1"})
