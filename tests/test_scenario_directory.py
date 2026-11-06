"""Tests for scenario directory discovery and aggregate validation."""

from __future__ import annotations

import json
from pathlib import Path

from proactive_bench.scenarios.directory import load_scenario_file, validate_directory


def write_scenario(root: Path, scenario_id: str, **overrides: object) -> Path:
    data: dict = {
        "id": scenario_id,
        "workspace": "backend",
        "need_type": "blockage",
        "persona": {"name": "Sam", "role": "backend engineer", "team": "payments"},
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
                "app": "repo",
                "function": "list_prs",
                "parameter": {},
            }
        ],
    }
    data.update(overrides)
    path = root / f"{scenario_id}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_scenario_file_valid(tmp_path: Path) -> None:
    path = write_scenario(tmp_path, "backend-alpha-01")
    loaded = load_scenario_file(path)
    assert loaded.errors == ()
    assert loaded.data["id"] == "backend-alpha-01"


def test_load_scenario_file_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    loaded = load_scenario_file(path)
    assert len(loaded.errors) == 1
    assert "JSON" in loaded.errors[0].message


def test_load_scenario_file_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    loaded = load_scenario_file(path)
    assert len(loaded.errors) == 1
    assert "object" in loaded.errors[0].message


def test_load_scenario_file_bad_encoding(tmp_path: Path) -> None:
    path = tmp_path / "utf16.json"
    path.write_bytes("id".encode("utf-16"))
    loaded = load_scenario_file(path)
    assert len(loaded.errors) == 1
    assert "cannot read file" in loaded.errors[0].message


def test_validate_directory_ok(tmp_path: Path) -> None:
    write_scenario(tmp_path, "backend-alpha-01")
    write_scenario(tmp_path, "backend-beta-02", workspace="backend", need_type="risk")
    report = validate_directory(tmp_path)
    assert report.ok
    assert report.scenario_count == 2
    assert report.global_errors == ()


def test_validate_directory_empty(tmp_path: Path) -> None:
    report = validate_directory(tmp_path)
    assert not report.ok
    assert any("no scenario" in e.message for e in report.global_errors)


def test_validate_directory_id_must_match_filename(tmp_path: Path) -> None:
    write_scenario(tmp_path, "orphan-id")
    # rewrite under wrong name
    src = tmp_path / "orphan-id.json"
    wrong = tmp_path / "renamed.json"
    wrong.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    src.unlink()
    report = validate_directory(tmp_path)
    assert not report.ok
    assert any("filename" in e.message or "stem" in e.message for e in report.files[0].errors)


def test_validate_directory_duplicate_ids(tmp_path: Path) -> None:
    write_scenario(tmp_path, "dup-id")
    # second file, same id inside, different name
    first = tmp_path / "dup-id.json"
    second = tmp_path / "other-name.json"
    second.write_text(first.read_text(encoding="utf-8"), encoding="utf-8")
    report = validate_directory(tmp_path)
    assert not report.ok
    assert any("duplicate" in e.message for e in report.global_errors)


def test_validate_directory_reports_schema_errors(tmp_path: Path) -> None:
    path = write_scenario(tmp_path, "bad-one")
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["need_type"]
    path.write_text(json.dumps(raw), encoding="utf-8")
    report = validate_directory(tmp_path)
    assert not report.ok
    assert report.scenario_count == 0
    assert any("need_type" in e.message for e in report.files[0].errors)
