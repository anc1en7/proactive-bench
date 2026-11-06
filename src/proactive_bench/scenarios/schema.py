"""Validate one scenario dict against the SPEC §3.5 schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from proactive_bench.env import (
    INITIAL_STATES,
    InvalidCommandError,
    MissingParameterError,
    ToolKindError,
    UnknownAppError,
    UnknownToolError,
    parse_command,
)

WORKSPACES: tuple[str, ...] = ("backend", "sre", "lead", "frontend", "platform")
NEED_TYPES: tuple[str, ...] = ("dependency", "blockage", "risk", "opportunity")

SCENARIO_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"id", "workspace", "need_type", "persona", "apps", "events"}
)
SCENARIO_OPTIONAL_KEYS: frozenset[str] = frozenset({"notes"})
_PERSONA_KEYS: frozenset[str] = frozenset({"name", "role", "team"})
_COMMAND_ERRORS = (
    InvalidCommandError,
    UnknownAppError,
    UnknownToolError,
    MissingParameterError,
    ToolKindError,
)


@dataclass(frozen=True)
class ScenarioError:
    """One validation failure, rooted at a file or in-memory source label."""

    source: str
    message: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


def validate_scenario(data: object, *, source: str = "<memory>") -> list[ScenarioError]:
    """Return every schema violation in ``data``; empty list means valid."""
    errors: list[ScenarioError] = []

    def add(message: str) -> None:
        errors.append(ScenarioError(source=source, message=message))

    if not isinstance(data, Mapping):
        add(f"scenario must be an object, got {type(data).__name__}")
        return errors

    missing = sorted(SCENARIO_REQUIRED_KEYS - set(data))
    if missing:
        add(f"missing required key(s): {', '.join(missing)}")
    unknown = sorted(set(data) - SCENARIO_REQUIRED_KEYS - SCENARIO_OPTIONAL_KEYS)
    if unknown:
        add(f"unknown key(s): {', '.join(unknown)}")

    scenario_id = data.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        add(f"id must be a non-empty string, got {scenario_id!r}")

    workspace = data.get("workspace")
    if workspace not in WORKSPACES:
        add(f"workspace must be one of {list(WORKSPACES)}, got {workspace!r}")

    need_type = data.get("need_type")
    if need_type not in NEED_TYPES:
        add(f"need_type must be one of {list(NEED_TYPES)}, got {need_type!r}")

    _validate_persona(data.get("persona"), add)
    _validate_apps(data.get("apps"), add)
    _validate_events(data.get("events"), add)

    if "notes" in data and not isinstance(data["notes"], str):
        add(f"notes must be a string, got {type(data['notes']).__name__}")

    return errors


def _validate_persona(persona: object, add: Any) -> None:
    if not isinstance(persona, Mapping):
        add(f"persona must be an object, got {type(persona).__name__}")
        return
    missing = sorted(_PERSONA_KEYS - set(persona))
    if missing:
        add(f"persona missing key(s): {', '.join(missing)}")
    for key in sorted(_PERSONA_KEYS & set(persona)):
        value = persona[key]
        if not isinstance(value, str) or not value:
            add(f"persona.{key} must be a non-empty string, got {value!r}")


def _validate_apps(apps: object, add: Any) -> None:
    if not isinstance(apps, Mapping):
        add(f"apps must be an object, got {type(apps).__name__}")
        return
    expected = set(INITIAL_STATES)
    actual = set(apps)
    for missing in sorted(expected - actual):
        add(f"apps missing app {missing!r}")
    for extra in sorted(actual - expected):
        add(f"apps has unknown app {extra!r}")
    for app in sorted(expected & actual):
        state = apps[app]
        if not isinstance(state, Mapping):
            add(f"apps.{app} must be an object, got {type(state).__name__}")
            continue
        want = set(INITIAL_STATES[app]())
        got = set(state)
        if want != got:
            add(
                f"apps.{app} top-level keys {sorted(got)} "
                f"do not match initial_state keys {sorted(want)}"
            )


def _validate_events(events: object, add: Any) -> None:
    if not isinstance(events, list):
        add(f"events must be a list, got {type(events).__name__}")
        return
    if not events:
        add("events must contain at least one event")
        return
    seen_ids: set[str] = set()
    for index, event in enumerate(events):
        label = f"events[{index}]"
        if not isinstance(event, Mapping):
            add(f"{label} must be an object, got {type(event).__name__}")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            add(f"{label}.event_id must be a non-empty string, got {event_id!r}")
        elif event_id in seen_ids:
            add(f"duplicate event_id {event_id!r}")
        else:
            seen_ids.add(event_id)
        time_raw = event.get("time")
        if not isinstance(time_raw, str):
            add(f"{label}.time must be an ISO-8601 string, got {time_raw!r}")
        else:
            try:
                datetime.fromisoformat(time_raw.replace("Z", "+00:00"))
            except ValueError:
                add(f"{label}.time is not ISO-8601: {time_raw!r}")
        try:
            parse_command(event)
        except _COMMAND_ERRORS as exc:
            add(f"{label} invalid command: {exc}")
