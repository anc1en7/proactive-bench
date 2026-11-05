"""Deterministic action-level classifier (SPEC §3.4): final tool + args → L0–L3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolError,
)
from proactive_bench.env.parser import Command
from proactive_bench.env.registry import get_tool

ActionLevel = Literal["L0", "L1", "L2", "L3"]

_LEVELS: dict[str, ActionLevel] = {
    "post_comment": "L1",
    "request_review": "L1",
    "retrigger_ci": "L1",
    "open_draft_pr": "L2",
    "merge_pr": "L3",
    "revert_commit": "L3",
    "send_message": "L1",
    "post_standup_update": "L1",
    "pin_message": "L1",
    "ack_alert": "L1",
    "page_oncall": "L1",
    "propose_rollback": "L2",
    "execute_rollback": "L3",
    "create_event": "L3",
    "decline_event": "L3",
    "propose_reschedule": "L2",
    "block_focus_time": "L2",
}


def classify_action_level(action: Command | Mapping[str, Any] | None) -> ActionLevel:
    """Map a final action to L0–L3.

    ``None`` (do nothing) and read-only tools are L0. Write tools use the
    deterministic ``_LEVELS`` table keyed by tool name; arguments are part of
    the input contract but no argument-dependent rules are active in v1.
    Raises ``UnknownAppError``/``UnknownToolError`` for unregistered tools.
    """
    if action is None:
        return "L0"
    if isinstance(action, Command):
        spec = get_tool(action.app, action.function)
    elif isinstance(action, Mapping):
        app = action.get("app")
        function = action.get("function")
        if not isinstance(app, str) or not isinstance(function, str):
            raise InvalidCommandError(
                "action must include string 'app' and 'function' keys"
            )
        spec = get_tool(app, function)
    else:
        raise InvalidCommandError(
            f"action must be a command or null, got {type(action).__name__}"
        )
    if spec.kind == "read":
        return "L0"
    level = _LEVELS.get(spec.name)
    if level is None:
        raise ToolError(
            f"write tool {spec.full_name} has no action-level mapping "
            "(add it to classifier._LEVELS)"
        )
    return level


def known_levels() -> tuple[ActionLevel, ...]:
    """All action levels, in order (reporting helper)."""
    return get_args(ActionLevel)
