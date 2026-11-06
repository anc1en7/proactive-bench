"""Tests for scenario schema validation (offline, inline dicts)."""

from __future__ import annotations

from proactive_bench.scenarios.schema import (
    NEED_TYPES,
    WORKSPACES,
    ScenarioError,
    validate_scenario,
)


def valid_scenario(**overrides: object) -> dict:
    base: dict = {
        "id": "backend-ci-freeze-01",
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
    base.update(overrides)
    return base


def messages(errors: list[ScenarioError]) -> str:
    return "\n".join(str(e) for e in errors)


def test_valid_scenario_has_no_errors() -> None:
    assert validate_scenario(valid_scenario()) == []


def test_workspaces_and_need_types_match_spec() -> None:
    assert WORKSPACES == ("backend", "sre", "lead", "frontend", "platform")
    assert NEED_TYPES == ("dependency", "blockage", "risk", "opportunity")


def test_rejects_non_dict() -> None:
    errors = validate_scenario(["not", "a", "dict"])
    assert len(errors) == 1
    assert "must be an object" in errors[0].message


def test_rejects_missing_required_keys() -> None:
    data = valid_scenario()
    del data["need_type"]
    del data["events"]
    errors = validate_scenario(data)
    text = messages(errors)
    assert "need_type" in text
    assert "events" in text


def test_rejects_unknown_top_level_key() -> None:
    errors = validate_scenario(valid_scenario(bonus=1))
    assert any("bonus" in e.message for e in errors)


def test_rejects_bad_workspace_and_need_type() -> None:
    errors = validate_scenario(valid_scenario(workspace="marketing", need_type="vibe"))
    text = messages(errors)
    assert "marketing" in text
    assert "vibe" in text


def test_rejects_bad_persona() -> None:
    errors = validate_scenario(valid_scenario(persona={"name": "Sam"}))
    text = messages(errors)
    assert "role" in text
    assert "team" in text


def test_rejects_missing_app() -> None:
    data = valid_scenario()
    del data["apps"]["pager"]
    errors = validate_scenario(data)
    assert any("pager" in e.message for e in errors)


def test_rejects_extra_app() -> None:
    data = valid_scenario()
    data["apps"]["email"] = {}
    errors = validate_scenario(data)
    assert any("email" in e.message for e in errors)


def test_rejects_app_state_with_wrong_top_level_keys() -> None:
    data = valid_scenario()
    data["apps"]["repo"] = {"prs": []}
    errors = validate_scenario(data)
    assert any("repo" in e.message for e in errors)


def test_rejects_empty_events() -> None:
    errors = validate_scenario(valid_scenario(events=[]))
    assert any("events" in e.message for e in errors)


def test_rejects_duplicate_event_ids() -> None:
    event = {
        "event_id": "e1",
        "time": "2026-11-10T09:00:00Z",
        "app": "repo",
        "function": "list_prs",
        "parameter": {},
    }
    errors = validate_scenario(valid_scenario(events=[event, dict(event)]))
    assert any("e1" in e.message for e in errors)


def test_rejects_bad_event_time() -> None:
    event = {
        "event_id": "e1",
        "time": "not-a-time",
        "app": "repo",
        "function": "list_prs",
        "parameter": {},
    }
    errors = validate_scenario(valid_scenario(events=[event]))
    assert any("time" in e.message for e in errors)


def test_rejects_event_unknown_tool() -> None:
    event = {
        "event_id": "e1",
        "time": "2026-11-10T09:00:00Z",
        "app": "repo",
        "function": "delete_everything",
        "parameter": {},
    }
    errors = validate_scenario(valid_scenario(events=[event]))
    assert any("delete_everything" in e.message for e in errors)


def test_rejects_event_missing_required_param() -> None:
    event = {
        "event_id": "e1",
        "time": "2026-11-10T09:00:00Z",
        "app": "repo",
        "function": "get_pr",
        "parameter": {},
    }
    errors = validate_scenario(valid_scenario(events=[event]))
    assert any("number" in e.message for e in errors)


def test_notes_optional_and_must_be_string() -> None:
    assert validate_scenario(valid_scenario(notes="construction notes")) == []
    errors = validate_scenario(valid_scenario(notes=42))
    assert any("notes" in e.message for e in errors)


def test_scenario_error_str_includes_source() -> None:
    err = ScenarioError(source="foo.json", message="boom")
    assert str(err) == "foo.json: boom"
