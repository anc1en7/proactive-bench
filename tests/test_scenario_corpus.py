"""Corpus tests: real scenarios/ files validate; Phase 3 coverage floors."""

from __future__ import annotations

from pathlib import Path

from proactive_bench.scenarios import NEED_TYPES, WORKSPACES, validate_directory

SCENARIOS_ROOT = Path(__file__).resolve().parent.parent / "scenarios"


def _report():
    return validate_directory(SCENARIOS_ROOT)


def test_scenarios_directory_exists() -> None:
    assert SCENARIOS_ROOT.is_dir()


def test_all_scenarios_validate() -> None:
    report = _report()
    assert report.global_errors == (), report.global_errors
    for scenario_file in report.files:
        assert scenario_file.errors == (), (
            f"{scenario_file.path.name}: {[str(e) for e in scenario_file.errors]}"
        )
    assert report.ok


def test_scenario_count_at_least_ten() -> None:
    assert _report().scenario_count >= 10


def test_every_workspace_represented_twice() -> None:
    report = _report()
    counts: dict[str, int] = {w: 0 for w in WORKSPACES}
    for scenario_file in report.files:
        if not scenario_file.errors:
            ws = scenario_file.data.get("workspace")
            if ws in counts:
                counts[ws] += 1
    assert all(v >= 2 for v in counts.values()), counts


def test_every_need_type_represented() -> None:
    report = _report()
    covered = {
        f.data.get("need_type") for f in report.files if not f.errors
    }
    assert set(NEED_TYPES) <= covered, f"missing: {set(NEED_TYPES) - covered}"
