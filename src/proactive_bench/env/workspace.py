"""Checkpoint workspace: deep-copy isolation plus validated tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from proactive_bench.env.base import ToolError, UnknownAppError
from proactive_bench.env.parser import Command
from proactive_bench.env.registry import APPS, INITIAL_STATES, get_tool


class Workspace:
    """Holds per-app state for one checkpoint.

    ``state()`` returns a live view (inspection/pre-seeding only); ``clone()``
    and ``snapshot()`` return deep copies. ``execute()`` never lets a handler
    result alias live state.
    """

    def __init__(self, states: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        provided = dict(states or {})
        for app in provided:
            if app not in APPS:
                known = ", ".join(sorted(APPS))
                raise UnknownAppError(f"unknown app {app!r} (known: {known})")
        self._states: dict[str, dict[str, Any]] = {}
        for app in APPS:
            if app in provided:
                self._states[app] = deepcopy(dict(provided[app]))
            else:
                self._states[app] = INITIAL_STATES[app]()

    def clone(self) -> Workspace:
        return Workspace(self._states)

    def state(self, app: str) -> dict[str, Any]:
        if app not in self._states:
            known = ", ".join(sorted(self._states))
            raise UnknownAppError(f"unknown app {app!r} (known: {known})")
        return self._states[app]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._states)

    def execute(self, command: Command) -> dict[str, Any]:
        spec = get_tool(command.app, command.function)
        spec.check_params(command.parameter)
        result = spec.handler(self.state(command.app), command.parameter)
        if not isinstance(result, dict):
            raise ToolError(
                f"{spec.full_name} handler returned {type(result).__name__}, expected dict"
            )
        return deepcopy(result)
