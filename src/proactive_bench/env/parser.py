"""Parse raw agent/event command dicts into validated Commands."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolKind,
    ToolKindError,
    ToolSpec,
)
from proactive_bench.env.registry import get_tool


@dataclass(frozen=True)
class Command:
    """One validated tool invocation: app id, function name, parameters."""

    app: str
    function: str
    parameter: dict[str, Any]


def parse_command(raw: Any) -> Command:
    """Validate a raw ``{app, function, parameter}`` mapping into a ``Command``.

    Extra keys (e.g. scenario ``event_id``/``time``) are ignored. A missing
    ``parameter`` defaults to ``{}``; ``null`` or non-object parameters raise.
    """
    if not isinstance(raw, Mapping):
        raise InvalidCommandError(
            f"command must be an object, got {type(raw).__name__}"
        )
    app = raw.get("app")
    function = raw.get("function")
    if not isinstance(app, str) or not app:
        raise InvalidCommandError(f"command.app must be a non-empty string, got {app!r}")
    if not isinstance(function, str) or not function:
        raise InvalidCommandError(
            f"command.function must be a non-empty string, got {function!r}"
        )
    params = raw.get("parameter", {})
    if not isinstance(params, Mapping):
        raise InvalidCommandError(
            f"command.parameter must be an object, got {type(params).__name__}"
        )
    spec = get_tool(app, function)
    copied = deepcopy(dict(params))
    spec.check_params(copied)
    return Command(app=app, function=function, parameter=copied)


def require_kind(command: Command, kind: ToolKind) -> ToolSpec:
    """Return the tool spec if ``command`` matches ``kind``, else raise."""
    spec = get_tool(command.app, command.function)
    if spec.kind != kind:
        raise ToolKindError(
            f"{spec.full_name} is a {spec.kind} tool, not {kind}"
        )
    return spec
