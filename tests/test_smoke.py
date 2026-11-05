"""Smoke tests: package imports and CLI basics (Phase 0 exit criteria)."""

from __future__ import annotations

import pytest

import proactive_bench
from proactive_bench.cli import build_parser, main


def test_version_exposed() -> None:
    assert proactive_bench.__version__ == "0.1.0"


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "proactive-bench" in out


def test_cli_unknown_command_not_implemented() -> None:
    assert main(["smoke"]) == 2


def test_parser_accepts_all_phase0_commands() -> None:
    parser = build_parser()
    for cmd in ("validate", "run", "report", "smoke"):
        args = parser.parse_args([cmd])
        assert args.command == cmd


def test_llm_public_api() -> None:
    from proactive_bench import llm

    for name in (
        "LLMConfigError",
        "LLMSettings",
        "OpenAICompatClient",
        "RateLimitError",
        "settings_from_env",
        "JSONExtractError",
        "JsonChatResult",
        "chat_json",
        "extract_json_text",
        "parse_json_text",
    ):
        assert hasattr(llm, name), name


def test_env_public_api() -> None:
    from proactive_bench import env

    for name in (
        "APPS",
        "ActionLevel",
        "Command",
        "InvalidCommandError",
        "MissingParameterError",
        "ToolError",
        "ToolExecutionError",
        "ToolKindError",
        "ToolSpec",
        "UnknownAppError",
        "UnknownToolError",
        "Workspace",
        "all_tools",
        "classify_action_level",
        "get_tool",
        "parse_command",
        "require_kind",
    ):
        assert hasattr(env, name), name
