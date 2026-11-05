"""Simulated apps: repo, chat, pager, cal — plus parser, isolation, classifier."""

from __future__ import annotations

from proactive_bench.env import cal, chat, pager, repo
from proactive_bench.env.base import (
    Handler,
    InvalidCommandError,
    MissingParameterError,
    ToolError,
    ToolExecutionError,
    ToolKind,
    ToolKindError,
    ToolSpec,
    UnknownAppError,
    UnknownToolError,
    as_int,
    next_id,
)
from proactive_bench.env.classifier import ActionLevel, classify_action_level, known_levels
from proactive_bench.env.parser import Command, parse_command, require_kind
from proactive_bench.env.registry import APPS, INITIAL_STATES, all_tools, get_tool, tool_kind
from proactive_bench.env.workspace import Workspace

__all__ = [
    "APPS",
    "ActionLevel",
    "Command",
    "Handler",
    "INITIAL_STATES",
    "InvalidCommandError",
    "MissingParameterError",
    "ToolError",
    "ToolExecutionError",
    "ToolKind",
    "ToolKindError",
    "ToolSpec",
    "UnknownAppError",
    "UnknownToolError",
    "Workspace",
    "all_tools",
    "as_int",
    "cal",
    "chat",
    "classify_action_level",
    "get_tool",
    "known_levels",
    "next_id",
    "pager",
    "parse_command",
    "repo",
    "require_kind",
    "tool_kind",
]
