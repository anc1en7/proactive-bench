"""Allowlist registry aggregating all four simulated apps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from proactive_bench.env import cal, chat, pager, repo
from proactive_bench.env.base import ToolKind, ToolSpec, UnknownAppError, UnknownToolError

APPS: dict[str, tuple[ToolSpec, ...]] = {
    "repo": repo.TOOLS,
    "chat": chat.TOOLS,
    "pager": pager.TOOLS,
    "cal": cal.TOOLS,
}

INITIAL_STATES: dict[str, Callable[[], dict[str, Any]]] = {
    "repo": repo.initial_state,
    "chat": chat.initial_state,
    "pager": pager.initial_state,
    "cal": cal.initial_state,
}


def get_tool(app: str, name: str) -> ToolSpec:
    """Look up one allowlisted tool; raise ``UnknownAppError``/``UnknownToolError``."""
    if app not in APPS:
        known = ", ".join(sorted(APPS))
        raise UnknownAppError(f"unknown app {app!r} (known: {known})")
    for spec in APPS[app]:
        if spec.name == name:
            return spec
    known = ", ".join(s.name for s in APPS[app])
    raise UnknownToolError(f"unknown tool {app}.{name} (known: {known})")


def all_tools() -> tuple[ToolSpec, ...]:
    return tuple(spec for specs in APPS.values() for spec in specs)


def tool_kind(app: str, name: str) -> ToolKind:
    return get_tool(app, name).kind
