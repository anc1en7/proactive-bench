"""Tests for `proactive-bench validate` (offline, tmp dirs)."""

from __future__ import annotations

import json
from pathlib import Path

from proactive_bench.cli import main


def write_ok(root: Path, scenario_id: str) -> None:
    data = {
        "id": scenario_id,
        "workspace": "sre",
        "need_type": "blockage",
        "persona": {"name": "Alex", "role": "SRE", "team": "infra"},
        "apps": {
            "repo": {"prs": [], "issues": [], "commits": []},
            "chat": {"channels": []},
            "pager": {
                "alerts": [],
                "runbooks": [],
                "deploy_windows": [],
                "rollback_plans": [],
                "oncall": {"primary": None, "paged": False, "page_log": []},
            },
            "cal": {"events": [], "focus_blocks": []},
        },
        "events": [
            {
                "event_id": "e1",
                "time": "2026-11-10T09:00:00Z",
                "app": "pager",
                "function": "list_alerts",
                "parameter": {},
            }
        ],
    }
    (root / f"{scenario_id}.json").write_text(json.dumps(data), encoding="utf-8")


def test_validate_ok_exits_zero(tmp_path: Path, capsys) -> None:
    write_ok(tmp_path, "sre-alert-01")
    code = main(["validate", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "OK sre-alert-01.json" in out
    assert "1 scenarios, 0 error(s)" in out


def test_validate_failure_exits_one(tmp_path: Path, capsys) -> None:
    (tmp_path / "bad.json").write_text("{}", encoding="utf-8")
    code = main(["validate", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL bad.json" in out
    assert "0 scenarios" in out


def test_validate_default_dir_is_scenarios(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scenarios").mkdir()
    write_ok(tmp_path / "scenarios", "backend-x-01")
    code = main(["validate"])
    assert code == 0
    assert "OK backend-x-01.json" in capsys.readouterr().out


def test_other_commands_still_unimplemented(capsys) -> None:
    assert main(["smoke"]) == 2
    assert main(["run"]) == 2
    assert main(["report"]) == 2
