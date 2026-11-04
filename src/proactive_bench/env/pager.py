"""Simulated on-call pager app: alerts, runbooks, deploy windows, rollbacks."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    next_id,
)


def initial_state() -> dict[str, Any]:
    return {
        "alerts": [],
        "runbooks": [],
        "deploy_windows": [],
        "rollback_plans": [],
        "oncall": {"primary": None, "paged": False, "page_log": []},
    }


def _require_id(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _find(collection: list[dict[str, Any]], value: str, label: str) -> dict[str, Any]:
    for item in collection:
        if item.get("id") == value:
            return item
    raise ToolExecutionError(f"no {label} with id {value!r}")


def _find_alert(state: dict[str, Any], alert_id: str) -> dict[str, Any]:
    return _find(state.setdefault("alerts", []), alert_id, "alert")


def _require_text(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


# --- reads -----------------------------------------------------------------


def _list_alerts(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"alerts": list(state.setdefault("alerts", []))}


def _get_alert(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"alert": _find_alert(state, _require_id(params, "id"))}


def _get_runbook(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    runbook = _find(state.setdefault("runbooks", []), _require_id(params, "id"), "runbook")
    return {"runbook": runbook}


def _get_deploy_window(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {
        "window": _find(
            state.setdefault("deploy_windows", []), _require_id(params, "id"), "deploy window"
        )
    }


# --- writes ----------------------------------------------------------------


def _ack_alert(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    alert = _find_alert(state, _require_id(params, "id"))
    if alert.get("status") != "firing":
        raise ToolExecutionError(
            f"alert {alert['id']} is {alert.get('status')!r}, not 'firing'"
        )
    alert["status"] = "acked"
    return {"ok": True, "alert": alert}


def _page_oncall(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    alert_id = _require_id(params, "alert_id")
    reason = _require_text(params, "reason")
    _find_alert(state, alert_id)
    oncall = state.setdefault("oncall", {"primary": None, "paged": False, "page_log": []})
    oncall["paged"] = True
    oncall.setdefault("page_log", []).append({"alert_id": alert_id, "reason": reason})
    return {"ok": True, "oncall": oncall}


def _propose_rollback(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    service = _require_text(params, "service")
    reason = _require_text(params, "reason")
    plans = state.setdefault("rollback_plans", [])
    plan = {"id": next_id(plans, "rb"), "service": service, "reason": reason, "status": "proposed"}
    plans.append(plan)
    return {"ok": True, "plan": plan}


def _execute_rollback(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    plans = state.setdefault("rollback_plans", [])
    plan = _find(plans, _require_id(params, "plan_id"), "rollback plan")
    if plan.get("status") != "proposed":
        raise ToolExecutionError(
            f"rollback plan {plan['id']} is {plan.get('status')!r}, not 'proposed'"
        )
    plan["status"] = "executed"
    return {"ok": True, "plan": plan}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="pager", name="list_alerts", kind="read", handler=_list_alerts),
    ToolSpec(app="pager", name="get_alert", kind="read", handler=_get_alert, required=("id",)),
    ToolSpec(
        app="pager", name="get_runbook", kind="read", handler=_get_runbook, required=("id",)
    ),
    ToolSpec(
        app="pager",
        name="get_deploy_window",
        kind="read",
        handler=_get_deploy_window,
        required=("id",),
    ),
    ToolSpec(app="pager", name="ack_alert", kind="write", handler=_ack_alert, required=("id",)),
    ToolSpec(
        app="pager",
        name="page_oncall",
        kind="write",
        handler=_page_oncall,
        required=("alert_id", "reason"),
    ),
    ToolSpec(
        app="pager",
        name="propose_rollback",
        kind="write",
        handler=_propose_rollback,
        required=("service", "reason"),
    ),
    ToolSpec(
        app="pager",
        name="execute_rollback",
        kind="write",
        handler=_execute_rollback,
        required=("plan_id",),
    ),
)
