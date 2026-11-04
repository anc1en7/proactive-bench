"""Simulated calendar app: events, declines, reschedule proposals, focus blocks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    as_int,
    next_id,
)

_WORKDAY_START_HOUR = 9
_WORKDAY_END_HOUR = 17
_SLOT_STEP_MINUTES = 30


def initial_state() -> dict[str, Any]:
    return {"events": [], "focus_blocks": []}


def _parse_ts(value: Any, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{name} must be an ISO-8601 string, got {value!r}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidCommandError(f"{name} is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _try_parse_ts(value: Any) -> datetime | None:
    try:
        return _parse_ts(value, "timestamp")
    except InvalidCommandError:
        return None


def _require_range(params: dict[str, Any]) -> tuple[datetime, datetime]:
    start = _parse_ts(params["start"], "start")
    end = _parse_ts(params["end"], "end")
    if end <= start:
        raise InvalidCommandError(f"end must be after start ({params['start']}..{params['end']})")
    return start, end


def _require_text(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


def _find_event(state: dict[str, Any], event_id: str) -> dict[str, Any]:
    for event in state.setdefault("events", []):
        if event.get("id") == event_id:
            return event
    raise ToolExecutionError(f"no event with id {event_id!r}")


def _require_id(params: dict[str, Any], key: str) -> str:
    value = params[key]
    if not isinstance(value, str) or not value:
        raise InvalidCommandError(f"{key} must be a non-empty string, got {value!r}")
    return value


# --- reads -----------------------------------------------------------------


def _list_events(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"events": list(state.setdefault("events", []))}


def _get_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"event": _find_event(state, _require_id(params, "id"))}


def _get_free_slots(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    date = params["date"]
    if not isinstance(date, str):
        raise InvalidCommandError(f"date must be YYYY-MM-DD, got {date!r}")
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as exc:
        raise InvalidCommandError(f"date must be YYYY-MM-DD, got {date!r}") from exc
    duration = as_int(params.get("duration_minutes", 60), "duration_minutes")
    if duration <= 0:
        raise InvalidCommandError(f"duration_minutes must be positive, got {duration}")
    day_start = datetime.fromisoformat(f"{date}T{_WORKDAY_START_HOUR:02d}:00:00+00:00")
    day_end = datetime.fromisoformat(f"{date}T{_WORKDAY_END_HOUR:02d}:00:00+00:00")
    busy: list[tuple[datetime, datetime]] = []
    for item in [*state.setdefault("events", []), *state.setdefault("focus_blocks", [])]:
        if item.get("status") == "declined":
            continue
        start = _try_parse_ts(item.get("start"))
        end = _try_parse_ts(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        if start < day_end and end > day_start:
            busy.append((max(start, day_start), min(end, day_end)))
    slot_delta = timedelta(minutes=duration)
    step = timedelta(minutes=_SLOT_STEP_MINUTES)
    slots: list[dict[str, str]] = []
    cursor = day_start
    while cursor + slot_delta <= day_end:
        slot_end = cursor + slot_delta
        if all(slot_end <= b0 or cursor >= b1 for b0, b1 in busy):
            slots.append({"start": cursor.isoformat(), "end": slot_end.isoformat()})
        cursor += step
    return {"slots": slots}


# --- writes ----------------------------------------------------------------


def _create_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = _require_text(params, "title")
    start, end = _require_range(params)
    attendees = params.get("attendees", [])
    if not isinstance(attendees, list):
        raise InvalidCommandError(f"attendees must be a list, got {attendees!r}")
    events = state.setdefault("events", [])
    event = {
        "id": next_id(events, "e"),
        "title": title,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "attendees": list(attendees),
        "status": "confirmed",
    }
    events.append(event)
    return {"ok": True, "event": event}


def _decline_event(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    event = _find_event(state, _require_id(params, "id"))
    if event.get("status") == "declined":
        raise ToolExecutionError(f"event {event['id']} already declined")
    event["status"] = "declined"
    return {"ok": True, "event": event}


def _propose_reschedule(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    event = _find_event(state, _require_id(params, "id"))
    if event.get("status") == "declined":
        raise ToolExecutionError(f"event {event['id']} is declined")
    start, end = _require_range(params)
    event["status"] = "reschedule_proposed"
    event["proposed_start"] = start.isoformat()
    event["proposed_end"] = end.isoformat()
    return {"ok": True, "event": event}


def _block_focus_time(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    title = _require_text(params, "title")
    start, end = _require_range(params)
    blocks = state.setdefault("focus_blocks", [])
    block = {"id": next_id(blocks, "f"), "title": title, "start": start.isoformat(),
             "end": end.isoformat()}
    blocks.append(block)
    return {"ok": True, "block": block}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="cal", name="list_events", kind="read", handler=_list_events),
    ToolSpec(app="cal", name="get_event", kind="read", handler=_get_event, required=("id",)),
    ToolSpec(
        app="cal", name="get_free_slots", kind="read", handler=_get_free_slots, required=("date",)
    ),
    ToolSpec(
        app="cal",
        name="create_event",
        kind="write",
        handler=_create_event,
        required=("title", "start", "end"),
    ),
    ToolSpec(
        app="cal", name="decline_event", kind="write", handler=_decline_event, required=("id",)
    ),
    ToolSpec(
        app="cal",
        name="propose_reschedule",
        kind="write",
        handler=_propose_reschedule,
        required=("id", "start", "end"),
    ),
    ToolSpec(
        app="cal",
        name="block_focus_time",
        kind="write",
        handler=_block_focus_time,
        required=("title", "start", "end"),
    ),
)
