"""Shared tool specification, error types, and helpers for simulated apps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

ToolKind = Literal["read", "write"]

Handler = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class ToolError(Exception):
    """Base class for simulated-app failures surfaced to the agent."""


class InvalidCommandError(ToolError):
    """The command payload is structurally invalid or has bad value types."""


class UnknownAppError(ToolError):
    """The requested app id is not part of the benchmark."""


class UnknownToolError(ToolError):
    """The requested function is not in the app's allowlist."""


class MissingParameterError(ToolError):
    """A required parameter was absent from the command."""


class ToolKindError(ToolError):
    """A read tool was used where a write is required, or vice versa."""


class ToolExecutionError(ToolError):
    """The tool ran but could not complete (e.g. entity not found)."""


@dataclass(frozen=True)
class ToolSpec:
    """One allowlisted tool: metadata plus its handler."""

    app: str
    name: str
    kind: ToolKind
    handler: Handler
    required: tuple[str, ...] = ()

    @property
    def full_name(self) -> str:
        return f"{self.app}.{self.name}"

    def check_params(self, params: Mapping[str, Any]) -> None:
        missing = [key for key in self.required if key not in params]
        if missing:
            raise MissingParameterError(
                f"{self.full_name} missing required parameter(s): {', '.join(missing)}"
            )


def next_id(items: list[dict[str, Any]], prefix: str) -> str:
    """Next ``<prefix><n>`` id, one past the max numeric suffix already present."""
    highest = 0
    for item in items:
        raw = str(item.get("id", ""))
        if raw.startswith(prefix) and raw[len(prefix) :].isdigit():
            highest = max(highest, int(raw[len(prefix) :]))
    return f"{prefix}{highest + 1}"


def as_int(value: Any, name: str) -> int:
    """Coerce a JSON integer or digit string to ``int``; else raise."""
    if isinstance(value, bool):
        raise InvalidCommandError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.removeprefix("-").isdecimal():
            return int(s)
    raise InvalidCommandError(f"{name} must be an integer, got {value!r}")
