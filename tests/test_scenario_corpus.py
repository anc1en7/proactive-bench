"""Corpus tests: real scenarios/ files validate; coverage floors hold."""

from __future__ import annotations

from pathlib import Path

from proactive_bench.scenarios import WORKSPACES, validate_directory

SCENARIOS_ROOT = Path(__file__).resolve().parent.parent / "scenarios"


def test_scenarios_directory_exists() -> None:
    assert SCENARIOS_ROOT.is_dir()


def test_all_scenarios_validate() -> None:
    report = validate_directory(SCENARIOS_ROOT)
    assert report.global_errors == (), report.global_errors
    for scenario_file in report.files:
        assert scenario_file.errors == (), (
            f"{scenario_file.path.name}: {[str(e) for e in scenario_file.errors]}"
        )
    assert report.ok


def test_scenario_count_at_least_five() -> None:
    report = validate_directory(SCENARIOS_ROOT)
    assert report.scenario_count >= 5


def test_every_workspace_represented() -> None:
    report = validate_directory(SCENARIOS_ROOT)
    covered = {f.data.get("workspace") for f in report.files if not f.errors}
    assert set(WORKSPACES) <= covered, f"missing workspaces: {set(WORKSPACES) - covered}"
