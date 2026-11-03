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
