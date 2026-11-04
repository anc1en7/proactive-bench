"""Tests for the simulated calendar app (offline)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from proactive_bench.env import cal
from proactive_bench.env.base import InvalidCommandError, ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in cal.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "events": [
            {"id": "e1", "title": "sync", "start": "2026-11-10T10:00:00+00:00",
             "end": "2026-11-10T11:00:00+00:00", "attendees": ["alice"],
             "status": "confirmed"},
            {"id": "e2", "title": "dental", "start": "2026-11-11T14:00:00+00:00",
             "end": "2026-11-11T15:00:00+00:00", "attendees": [],
             "status": "declined"},
        ],
        "focus_blocks": [
            {"id": "f1", "title": "deep work", "start": "2026-11-10T13:00:00+00:00",
             "end": "2026-11-10T14:00:00+00:00"},
        ],
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in cal.TOOLS if s.kind == "read"}
    writes = {s.name for s in cal.TOOLS if s.kind == "write"}
    assert reads == {"list_events", "get_event", "get_free_slots"}
    assert writes == {"create_event", "decline_event", "propose_reschedule", "block_focus_time"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = cal.initial_state(), cal.initial_state()
    assert a == {"events": [], "focus_blocks": []}
    a["events"].append(1)
    assert b["events"] == []


def test_list_and_get_events() -> None:
    state = seed_state()
    assert len(get("list_events").handler(state, {})["events"]) == 2
    assert get("get_event").handler(state, {"id": "e1"})["event"]["title"] == "sync"
    with pytest.raises(ToolExecutionError):
        get("get_event").handler(state, {"id": "e9"})


def test_create_event_assigns_id_and_validates_range() -> None:
    state = seed_state()
    result = get("create_event").handler(
        state,
        {"title": "1:1", "start": "2026-11-12T09:00:00Z", "end": "2026-11-12T09:30:00Z",
         "attendees": ["bob"]},
    )
    event = result["event"]
    assert event["id"] == "e3"
    assert event["status"] == "confirmed"
    assert event["start"] == "2026-11-12T09:00:00+00:00"
    with pytest.raises(InvalidCommandError, match="after start"):
        get("create_event").handler(
            state,
            {"title": "bad", "start": "2026-11-12T10:00:00Z", "end": "2026-11-12T09:00:00Z"},
        )
    with pytest.raises(InvalidCommandError, match="ISO-8601"):
        get("create_event").handler(
            state, {"title": "bad", "start": "not-a-time", "end": "2026-11-12T09:00:00Z"}
        )


def test_create_event_rejects_non_list_attendees() -> None:
    with pytest.raises(InvalidCommandError, match="attendees"):
        get("create_event").handler(
            seed_state(),
            {"title": "x", "start": "2026-11-12T09:00:00Z", "end": "2026-11-12T10:00:00Z",
             "attendees": "alice"},
        )


def test_decline_event_transitions_once() -> None:
    state = seed_state()
    get("decline_event").handler(state, {"id": "e1"})
    assert state["events"][0]["status"] == "declined"
    with pytest.raises(ToolExecutionError, match="already declined"):
        get("decline_event").handler(state, {"id": "e1"})


def test_propose_reschedule_sets_proposed_fields() -> None:
    state = seed_state()
    get("propose_reschedule").handler(
        state,
        {"id": "e1", "start": "2026-11-10T15:00:00Z", "end": "2026-11-10T16:00:00Z"},
    )
    event = state["events"][0]
    assert event["status"] == "reschedule_proposed"
    assert event["proposed_start"] == "2026-11-10T15:00:00+00:00"
    assert event["proposed_end"] == "2026-11-10T16:00:00+00:00"
    with pytest.raises(ToolExecutionError, match="declined"):
        get("propose_reschedule").handler(
            state, {"id": "e2", "start": "2026-11-11T16:00:00Z", "end": "2026-11-11T17:00:00Z"}
        )


def test_block_focus_time_appends_block() -> None:
    state = seed_state()
    result = get("block_focus_time").handler(
        state,
        {"title": "write spec", "start": "2026-11-13T09:00:00Z", "end": "2026-11-13T11:00:00Z"},
    )
    assert result["block"]["id"] == "f2"
    assert len(state["focus_blocks"]) == 2


def test_get_free_slots_skips_busy_and_declined() -> None:
    state = seed_state()
    result = get("get_free_slots").handler(state, {"date": "2026-11-10"})
    slots = result["slots"]
    starts = [s["start"] for s in slots]
    # Busy: event 10:00-11:00 blocks 09:30/10:00/10:30 candidates; focus 13:00-14:00
    # blocks 12:30/13:00/13:30 candidates (60-min slots).
    assert "2026-11-10T09:00:00+00:00" in starts
    assert "2026-11-10T11:00:00+00:00" in starts
    assert "2026-11-10T14:00:00+00:00" in starts
    for blocked in (
        "2026-11-10T09:30:00+00:00",
        "2026-11-10T10:00:00+00:00",
        "2026-11-10T10:30:00+00:00",
        "2026-11-10T12:30:00+00:00",
        "2026-11-10T13:00:00+00:00",
        "2026-11-10T13:30:00+00:00",
    ):
        assert blocked not in starts
    assert starts[0] == "2026-11-10T09:00:00+00:00"
    assert starts[-1] == "2026-11-10T16:00:00+00:00"
    # 15 candidates in 09:00..16:00 step 30, minus 6 blocked = 9.
    assert len(slots) == 9
    # Declined event on 11-11 does not block that day.
    day2 = get("get_free_slots").handler(state, {"date": "2026-11-11"})["slots"]
    assert len(day2) == 15


def test_get_free_slots_honors_duration_and_validates_date() -> None:
    state = seed_state()
    slots = get("get_free_slots").handler(
        state, {"date": "2026-11-10", "duration_minutes": 120}
    )["slots"]
    for slot in slots:
        assert datetime.fromisoformat(slot["end"]) - datetime.fromisoformat(slot["start"]) == (
            timedelta(minutes=120)
        )
    with pytest.raises(InvalidCommandError, match="YYYY-MM-DD"):
        get("get_free_slots").handler(state, {"date": "10-11-2026"})
    with pytest.raises(InvalidCommandError, match="duration_minutes"):
        get("get_free_slots").handler(state, {"date": "2026-11-10", "duration_minutes": 0})


def test_get_free_slots_accepts_non_padded_date() -> None:
    slots = get("get_free_slots").handler(seed_state(), {"date": "2026-1-1"})["slots"]
    starts = [s["start"] for s in slots]
    assert starts[0] == "2026-01-01T09:00:00+00:00"
    assert starts[-1] == "2026-01-01T16:00:00+00:00"
    assert len(slots) == 15
    for slot in slots:
        assert slot["start"].endswith("+00:00")
        assert slot["end"].endswith("+00:00")


def test_get_free_slots_rejects_huge_duration() -> None:
    with pytest.raises(InvalidCommandError, match="duration_minutes"):
        get("get_free_slots").handler(
            seed_state(), {"date": "2026-11-10", "duration_minutes": 6_000_000_000}
        )
