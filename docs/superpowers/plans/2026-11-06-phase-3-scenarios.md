# Phase 3 — Scenarios Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the scenario schema validator (`proactive-bench validate`), five workspace personas, and ≥10 offline scenario JSON files that pass validation — Phase 3 exit criteria from SPEC §11.

**Architecture:** A `scenarios` package holds pure validation logic (`validate_scenario` for one dict, `validate_directory` for `scenarios/*.json`). The CLI `validate` subcommand wires it up and exits non-zero on errors. Scenario data lives in top-level `scenarios/` as self-contained JSON (persona + full four-app initial state + timed events), matching SPEC §3.5 exactly — no shared-fixture indirection.

**Tech Stack:** Python 3.11+, stdlib only (`json`, `pathlib`, `datetime`, `dataclasses`, `typing`), pytest, ruff. Reuses Phase 2 `env` (`APPS`, `INITIAL_STATES`, `parse_command`).

**Spec:** [`SPEC.md`](../../../SPEC.md) — §3.2 (workspaces), §3.3 (need types), §3.5 (scenario schema + checkpoints), §9 (scenario test layer), §11 Phase 3 row.

## Global Constraints

- Python 3.11+; ruff `select = ["E", "F", "I", "UP", "B"]`, `line-length = 100` (pyproject.toml).
- Tests run fully offline — no network (SPEC §9).
- No new runtime dependencies (stdlib `json`/`pathlib` only).
- Commit message style: `Phase 3: <component>` (matches Phase 0–2 history).
- Scenario schema (SPEC §3.5) — required top-level keys: `id`, `workspace`, `need_type`, `persona`, `apps`, `events`; optional: `notes`.
- `workspace` ∈ `backend | sre | lead | frontend | platform` (SPEC §3.2).
- `need_type` ∈ `dependency | blockage | risk | opportunity` (SPEC §3.3).
- `persona` required keys: `name`, `role`, `team` (all non-empty strings).
- `apps` must contain exactly the four app ids `repo`, `chat`, `pager`, `cal`; each value a dict whose top-level keys equal `INITIAL_STATES[app]()` keys (SPEC §3.1 shapes).
- Each event: `event_id` (unique within scenario), `time` (ISO-8601 parseable), `app`, `function`, `parameter` — `app`/`function`/`parameter` must pass `parse_command` (allowlist + required params). Events may be reads or writes (they set up the world).
- Scenario `id` unique across the directory; file name must be `{id}.json`.
- ≥10 scenario files at Phase 3 exit; all five workspaces represented; all four need types represented (SPEC §11 + §3.2 balance).
- `proactive-bench validate` exits 0 when every scenario passes, 1 when any fail (Phase 3 exit criterion).
- Secrets never committed (n/a; `.env` stays gitignored).

---

### Task 1: `scenarios/schema.py` — validate one scenario

**Files:**
- Create: `src/proactive_bench/scenarios/__init__.py`
- Create: `src/proactive_bench/scenarios/schema.py`
- Test: `tests/test_scenario_schema.py`

**Interfaces:**
- Consumes: `proactive_bench.env` — `APPS`, `INITIAL_STATES`, `parse_command`, `InvalidCommandError`, `UnknownAppError`, `UnknownToolError`, `MissingParameterError`, `ToolKindError`.
- Produces (used by later tasks):
  - `WORKSPACES: tuple[str, ...] = ("backend", "sre", "lead", "frontend", "platform")`
  - `NEED_TYPES: tuple[str, ...] = ("dependency", "blockage", "risk", "opportunity")`
  - `@dataclass(frozen=True) class ScenarioError(path: str, message: str)` with `__str__` → `f"{path}: {message}"`
  - `validate_scenario(data: object, *, source: str = "<memory>") -> list[ScenarioError]` — empty list means valid
  - `SCENARIO_REQUIRED_KEYS: frozenset[str] = frozenset({"id", "workspace", "need_type", "persona", "apps", "events"})`
  - `SCENARIO_OPTIONAL_KEYS: frozenset[str] = frozenset({"notes"})`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenario_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.scenarios'`

- [ ] **Step 3: Write minimal implementation**

`src/proactive_bench/scenarios/__init__.py`:

```python
"""Scenario schema validation (SPEC §3.5)."""

from __future__ import annotations

from proactive_bench.scenarios.schema import (
    NEED_TYPES,
    SCENARIO_OPTIONAL_KEYS,
    SCENARIO_REQUIRED_KEYS,
    WORKSPACES,
    ScenarioError,
    validate_scenario,
)

__all__ = [
    "NEED_TYPES",
    "SCENARIO_OPTIONAL_KEYS",
    "SCENARIO_REQUIRED_KEYS",
    "WORKSPACES",
    "ScenarioError",
    "validate_scenario",
]
```

`src/proactive_bench/scenarios/schema.py`:

```python
"""Validate one scenario dict against the SPEC §3.5 schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from proactive_bench.env import (
    INITIAL_STATES,
    InvalidCommandError,
    MissingParameterError,
    ToolKindError,
    UnknownAppError,
    UnknownToolError,
    parse_command,
)

WORKSPACES: tuple[str, ...] = ("backend", "sre", "lead", "frontend", "platform")
NEED_TYPES: tuple[str, ...] = ("dependency", "blockage", "risk", "opportunity")

SCENARIO_REQUIRED_KEYS: frozenset[str] = frozenset(
    {"id", "workspace", "need_type", "persona", "apps", "events"}
)
SCENARIO_OPTIONAL_KEYS: frozenset[str] = frozenset({"notes"})
_PERSONA_KEYS: frozenset[str] = frozenset({"name", "role", "team"})
_COMMAND_ERRORS = (
    InvalidCommandError,
    UnknownAppError,
    UnknownToolError,
    MissingParameterError,
    ToolKindError,
)


@dataclass(frozen=True)
class ScenarioError:
    """One validation failure, rooted at a file or in-memory source label."""

    source: str
    message: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


def validate_scenario(data: object, *, source: str = "<memory>") -> list[ScenarioError]:
    """Return every schema violation in ``data``; empty list means valid."""
    errors: list[ScenarioError] = []

    def add(message: str) -> None:
        errors.append(ScenarioError(source=source, message=message))

    if not isinstance(data, Mapping):
        add(f"scenario must be an object, got {type(data).__name__}")
        return errors

    missing = sorted(SCENARIO_REQUIRED_KEYS - set(data))
    if missing:
        add(f"missing required key(s): {', '.join(missing)}")
    unknown = sorted(set(data) - SCENARIO_REQUIRED_KEYS - SCENARIO_OPTIONAL_KEYS)
    if unknown:
        add(f"unknown key(s): {', '.join(unknown)}")

    scenario_id = data.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        add(f"id must be a non-empty string, got {scenario_id!r}")

    workspace = data.get("workspace")
    if workspace not in WORKSPACES:
        add(f"workspace must be one of {list(WORKSPACES)}, got {workspace!r}")

    need_type = data.get("need_type")
    if need_type not in NEED_TYPES:
        add(f"need_type must be one of {list(NEED_TYPES)}, got {need_type!r}")

    _validate_persona(data.get("persona"), add)
    _validate_apps(data.get("apps"), add)
    _validate_events(data.get("events"), add)

    if "notes" in data and not isinstance(data["notes"], str):
        add(f"notes must be a string, got {type(data['notes']).__name__}")

    return errors


def _validate_persona(persona: object, add: Any) -> None:
    if not isinstance(persona, Mapping):
        add(f"persona must be an object, got {type(persona).__name__}")
        return
    missing = sorted(_PERSONA_KEYS - set(persona))
    if missing:
        add(f"persona missing key(s): {', '.join(missing)}")
    for key in sorted(_PERSONA_KEYS & set(persona)):
        value = persona[key]
        if not isinstance(value, str) or not value:
            add(f"persona.{key} must be a non-empty string, got {value!r}")


def _validate_apps(apps: object, add: Any) -> None:
    if not isinstance(apps, Mapping):
        add(f"apps must be an object, got {type(apps).__name__}")
        return
    expected = set(INITIAL_STATES)
    actual = set(apps)
    for missing in sorted(expected - actual):
        add(f"apps missing app {missing!r}")
    for extra in sorted(actual - expected):
        add(f"apps has unknown app {extra!r}")
    for app in sorted(expected & actual):
        state = apps[app]
        if not isinstance(state, Mapping):
            add(f"apps.{app} must be an object, got {type(state).__name__}")
            continue
        want = set(INITIAL_STATES[app]())
        got = set(state)
        if want != got:
            add(
                f"apps.{app} top-level keys {sorted(got)} "
                f"do not match initial_state keys {sorted(want)}"
            )


def _validate_events(events: object, add: Any) -> None:
    if not isinstance(events, list):
        add(f"events must be a list, got {type(events).__name__}")
        return
    if not events:
        add("events must contain at least one event")
        return
    seen_ids: set[str] = set()
    for index, event in enumerate(events):
        label = f"events[{index}]"
        if not isinstance(event, Mapping):
            add(f"{label} must be an object, got {type(event).__name__}")
            continue
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            add(f"{label}.event_id must be a non-empty string, got {event_id!r}")
        elif event_id in seen_ids:
            add(f"duplicate event_id {event_id!r}")
        else:
            seen_ids.add(event_id)
        time_raw = event.get("time")
        if not isinstance(time_raw, str):
            add(f"{label}.time must be an ISO-8601 string, got {time_raw!r}")
        else:
            try:
                datetime.fromisoformat(time_raw.replace("Z", "+00:00"))
            except ValueError:
                add(f"{label}.time is not ISO-8601: {time_raw!r}")
        try:
            parse_command(event)
        except _COMMAND_ERRORS as exc:
            add(f"{label} invalid command: {exc}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scenario_schema.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/scenarios/__init__.py src/proactive_bench/scenarios/schema.py tests/test_scenario_schema.py
git commit -m "Phase 3: scenario schema validator"
```

---

### Task 2: `scenarios/directory.py` — discover and validate `scenarios/*.json`

**Files:**
- Create: `src/proactive_bench/scenarios/directory.py`
- Test: `tests/test_scenario_directory.py`

**Interfaces:**
- Consumes: `validate_scenario`, `ScenarioError` from Task 1.
- Produces (used by Task 3+):
  - `@dataclass(frozen=True) class ScenarioFile(path: Path, data: dict[str, Any], errors: tuple[ScenarioError, ...])`
  - `@dataclass(frozen=True) class DirectoryReport(files: tuple[ScenarioFile, ...], global_errors: tuple[ScenarioError, ...])`
    - property `ok: bool` — no errors anywhere
    - property `scenario_count: int` — files with zero errors
  - `load_scenario_file(path: Path) -> ScenarioFile` — JSON parse errors become a single `ScenarioError`; non-object JSON → error
  - `validate_directory(root: Path) -> DirectoryReport`
    - discovers `root.glob("*.json")` (non-recursive; workspace subdirs are not scenarios)
    - per-file: JSON load → `validate_scenario(data, source=path.name)` → `id` must equal `path.stem`
    - global: duplicate `id` values across files; zero files → one global error `"no scenario JSON files found"`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenario_directory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'proactive_bench.scenarios.directory'`

- [ ] **Step 3: Write minimal implementation**

`src/proactive_bench/scenarios/directory.py`:

```python
"""Discover scenario JSON files under a root and validate them as a set."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from proactive_bench.scenarios.schema import ScenarioError, validate_scenario


@dataclass(frozen=True)
class ScenarioFile:
    """One scenario file after load + schema validation."""

    path: Path
    data: dict[str, Any]
    errors: tuple[ScenarioError, ...]


@dataclass(frozen=True)
class DirectoryReport:
    """Aggregate result for a scenarios root directory."""

    files: tuple[ScenarioFile, ...]
    global_errors: tuple[ScenarioError, ...]

    @property
    def ok(self) -> bool:
        return not self.global_errors and all(not f.errors for f in self.files)

    @property
    def scenario_count(self) -> int:
        return sum(1 for f in self.files if not f.errors)


def load_scenario_file(path: Path) -> ScenarioFile:
    """Read and JSON-parse one scenario file; never raises for bad content."""
    source = path.name
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return ScenarioFile(
            path=path,
            data={},
            errors=(ScenarioError(source=source, message=f"cannot read file: {exc}"),),
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ScenarioFile(
            path=path,
            data={},
            errors=(ScenarioError(source=source, message=f"invalid JSON: {exc}"),),
        )
    if not isinstance(data, dict):
        return ScenarioFile(
            path=path,
            data={},
            errors=(
                ScenarioError(
                    source=source,
                    message=f"scenario must be a JSON object, got {type(data).__name__}",
                ),
            ),
        )
    errors = tuple(validate_scenario(data, source=source))
    scenario_id = data.get("id")
    if isinstance(scenario_id, str) and scenario_id and scenario_id != path.stem:
        errors = (
            *errors,
            ScenarioError(
                source=source,
                message=(
                    f"id {scenario_id!r} does not match filename stem {path.stem!r}"
                ),
            ),
        )
    return ScenarioFile(path=path, data=data, errors=errors)


def validate_directory(root: Path) -> DirectoryReport:
    """Validate every ``*.json`` directly under ``root`` plus cross-file rules."""
    global_errors: list[ScenarioError] = []
    paths = sorted(root.glob("*.json"))
    if not paths:
        global_errors.append(
            ScenarioError(source=str(root), message="no scenario JSON files found")
        )
    files = tuple(load_scenario_file(path) for path in paths)
    seen: dict[str, Path] = {}
    for scenario_file in files:
        scenario_id = scenario_file.data.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            continue
        if scenario_id in seen:
            global_errors.append(
                ScenarioError(
                    source=scenario_file.path.name,
                    message=(
                        f"duplicate id {scenario_id!r} "
                        f"(also in {seen[scenario_id].name})"
                    ),
                )
            )
        else:
            seen[scenario_id] = scenario_file.path
    return DirectoryReport(files=files, global_errors=tuple(global_errors))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scenario_directory.py -v` → PASS
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/scenarios/directory.py tests/test_scenario_directory.py
git commit -m "Phase 3: scenario directory discovery"
```

---

### Task 3: CLI `validate` subcommand

**Files:**
- Modify: `src/proactive_bench/scenarios/__init__.py` (re-export directory API)
- Modify: `src/proactive_bench/cli.py`
- Test: `tests/test_cli_validate.py`

**Interfaces:**
- Consumes: `validate_directory`, `DirectoryReport` from Task 2; `config.yaml` key `scenarios_dir` (default `scenarios`).
- Produces:
  - `main(["validate"])` → prints one line per file (`OK <file>` / `FAIL <file>`), then `"<N> scenarios, <M> error(s)"`, exits **0** if `report.ok` else **1**
  - `main(["validate", "--dir", "path"])` → validates that directory instead (for tests)
  - Other subcommands (`run`/`report`/`smoke`) unchanged (still exit 2)

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_validate.py -v`
Expected: FAIL — `validate` still prints "not implemented yet" and returns 2 (or argparse rejects `--dir`)

- [ ] **Step 3: Write implementation**

Update `src/proactive_bench/scenarios/__init__.py` to also export directory API:

```python
"""Scenario schema validation (SPEC §3.5)."""

from __future__ import annotations

from proactive_bench.scenarios.directory import (
    DirectoryReport,
    ScenarioFile,
    load_scenario_file,
    validate_directory,
)
from proactive_bench.scenarios.schema import (
    NEED_TYPES,
    SCENARIO_OPTIONAL_KEYS,
    SCENARIO_REQUIRED_KEYS,
    WORKSPACES,
    ScenarioError,
    validate_scenario,
)

__all__ = [
    "DirectoryReport",
    "NEED_TYPES",
    "SCENARIO_OPTIONAL_KEYS",
    "SCENARIO_REQUIRED_KEYS",
    "ScenarioError",
    "ScenarioFile",
    "WORKSPACES",
    "load_scenario_file",
    "validate_directory",
    "validate_scenario",
]
```

Replace `src/proactive_bench/cli.py` with:

```python
"""CLI entrypoint: validate | run | report | smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from proactive_bench import __version__
from proactive_bench.scenarios import validate_directory

_COMMANDS = ("validate", "run", "report", "smoke")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proactive-bench",
        description="Mini benchmark for proactive engineering agents (HALO-lite).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")
    validate = sub.add_parser("validate", help="validate scenario JSON files")
    validate.add_argument(
        "--dir",
        type=Path,
        default=Path("scenarios"),
        help="scenarios root directory (default: scenarios)",
    )
    for name in _COMMANDS:
        if name != "validate":
            sub.add_parser(name, help=f"{name} (implemented in later phases)")
    return parser


def _run_validate(scenarios_dir: Path) -> int:
    report = validate_directory(scenarios_dir)
    for scenario_file in report.files:
        if scenario_file.errors:
            print(f"FAIL {scenario_file.path.name}")
            for error in scenario_file.errors:
                print(f"  {error}")
        else:
            print(f"OK {scenario_file.path.name}")
    for error in report.global_errors:
        print(f"FAIL {error}")
    error_count = sum(len(f.errors) for f in report.files) + len(report.global_errors)
    print(f"{report.scenario_count} scenarios, {error_count} error(s)")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "validate":
        return _run_validate(args.dir)
    print(
        f"command {args.command!r} is not implemented yet "
        f"(see SPEC.md phases; currently Phase 3).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_validate.py tests/test_smoke.py -v` → PASS
(`test_cli_unknown_command_not_implemented` still uses `smoke` → 2)
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add src/proactive_bench/scenarios/__init__.py src/proactive_bench/cli.py tests/test_cli_validate.py
git commit -m "Phase 3: CLI validate subcommand"
```

---

### Task 4: coverage guard + first five scenarios (one per workspace)

**Files:**
- Create: `scenarios/backend-ci-freeze-01.json`
- Create: `scenarios/sre-latency-page-01.json`
- Create: `scenarios/lead-review-queue-01.json`
- Create: `scenarios/frontend-release-train-01.json`
- Create: `scenarios/platform-cve-merge-01.json`
- Test: `tests/test_scenario_corpus.py`

**Interfaces:**
- Consumes: `validate_directory` (Task 2); Phase 2 `Workspace`/`parse_command` for optional replay checks.
- Produces: five valid scenario files; corpus test asserting directory `ok`, `scenario_count >= 5`, and every workspace appears at least once.

**Scenario authoring rules (all tasks):**
- Compact but coherent app state: enough PRs/alerts/events for the story, not exhaustive fixtures.
- Events must be allowlisted tools with required params (validator enforces).
- Event times strictly increasing (`e1` before `e2` …) — not schema-enforced; authoring rule for Phase 4 prefix logic.
- `notes` holds construction intent (never shown to agent).
- Commit message per file batch: `Phase 3: initial workspace scenarios`.

- [ ] **Step 1: Write the failing corpus test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenario_corpus.py -v`
Expected: FAIL — `no scenario JSON files found` (empty corpus)

- [ ] **Step 3: Write the five scenario JSON files**

`scenarios/backend-ci-freeze-01.json`:

```json
{
  "id": "backend-ci-freeze-01",
  "workspace": "backend",
  "need_type": "blockage",
  "persona": { "name": "Sam Okonkwo", "role": "backend engineer", "team": "payments" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 142,
          "title": "fix flaky ledger reconciliation test",
          "branch": "fix-reconcile-flake",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [
            { "name": "ci/unit", "status": "fail" },
            { "name": "ci/integration", "status": "pass" }
          ],
          "reviewers": ["priya"],
          "comments": [],
          "author": "sam"
        }
      ],
      "issues": [
        {
          "number": 88,
          "title": "Freeze window starts Friday 18:00 UTC",
          "state": "open",
          "labels": ["process"],
          "comments": []
        }
      ],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "payments-eng",
          "messages": [
            {
              "id": "m1",
              "author": "priya",
              "text": "Code freeze for the holiday sale starts Friday 18:00 UTC.",
              "pinned": true
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [
        {
          "id": "e1",
          "title": "Holiday sale freeze begins",
          "start": "2026-11-13T18:00:00+00:00",
          "end": "2026-11-16T09:00:00+00:00",
          "attendees": [],
          "status": "confirmed"
        }
      ],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-12T09:00:00Z",
      "app": "repo",
      "function": "get_checks",
      "parameter": { "number": 142 }
    },
    {
      "event_id": "e2",
      "time": "2026-11-12T09:05:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "payments-eng", "message_id": "m1" }
    }
  ],
  "notes": "Blockage: CI red two days before freeze; agent should retrigger or escalate, not force-merge."
}
```

`scenarios/sre-latency-page-01.json`:

```json
{
  "id": "sre-latency-page-01",
  "workspace": "sre",
  "need_type": "blockage",
  "persona": { "name": "Alex Rivera", "role": "site reliability engineer", "team": "edge" },
  "apps": {
    "repo": {
      "prs": [],
      "issues": [],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "sre-oncall",
          "messages": [
            {
              "id": "m1",
              "author": "alex",
              "text": "p99 latency on edge-gateway is 2.4s, SLO is 800ms",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [
        {
          "id": "a1",
          "severity": "sev2",
          "service": "edge-gateway",
          "status": "firing",
          "summary": "p99 latency 2.4s (SLO 800ms)"
        }
      ],
      "runbooks": [
        {
          "id": "rb-latency",
          "title": "Edge latency triage",
          "steps": ["check recent deploys", "check upstream error rate", "page primary if customer impact"]
        }
      ],
      "deploy_windows": [
        {
          "id": "dw1",
          "start": "2026-11-12T14:00:00Z",
          "end": "2026-11-12T16:00:00Z",
          "frozen": false
        }
      ],
      "rollback_plans": [],
      "oncall": { "primary": "alex", "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-12T14:10:00Z",
      "app": "pager",
      "function": "get_alert",
      "parameter": { "id": "a1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-12T14:11:00Z",
      "app": "pager",
      "function": "get_runbook",
      "parameter": { "id": "rb-latency" }
    }
  ],
  "notes": "Blockage: sev2 firing mid-deploy-window; ack/page/rollback are the judgment ladder."
}
```

`scenarios/lead-review-queue-01.json`:

```json
{
  "id": "lead-review-queue-01",
  "workspace": "lead",
  "need_type": "dependency",
  "persona": { "name": "Jordan Lee", "role": "tech lead", "team": "platform-experience" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 55,
          "title": "add audit log export API",
          "branch": "feat-audit-export",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [{ "name": "ci", "status": "pass" }],
          "reviewers": [],
          "comments": [],
          "author": "kai"
        },
        {
          "number": 56,
          "title": "migrate settings to v2 schema",
          "branch": "chore-settings-v2",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [{ "name": "ci", "status": "pass" }],
          "reviewers": ["jordan"],
          "comments": [],
          "author": "noor"
        }
      ],
      "issues": [],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "team-lead",
          "messages": [
            {
              "id": "m1",
              "author": "kai",
              "text": "@jordan blocked on review for #55 — release train cuts Wednesday",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [
        {
          "id": "e1",
          "title": "Release train cut",
          "start": "2026-11-18T15:00:00+00:00",
          "end": "2026-11-18T16:00:00+00:00",
          "attendees": ["jordan", "kai", "noor"],
          "status": "confirmed"
        }
      ],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-16T10:00:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "team-lead", "message_id": "m1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-16T10:02:00Z",
      "app": "repo",
      "function": "get_pr",
      "parameter": { "number": 55 }
    }
  ],
  "notes": "Dependency: lead is the blocker for Kai's PR before release train; request_review/post_comment vs wait."
}
```

`scenarios/frontend-release-train-01.json`:

```json
{
  "id": "frontend-release-train-01",
  "workspace": "frontend",
  "need_type": "risk",
  "persona": { "name": "Morgan Blake", "role": "frontend engineer", "team": "web-ui" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 201,
          "title": "checkout modal focus trap",
          "branch": "fix-modal-focus",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [
            { "name": "ci", "status": "pass" },
            { "name": "a11y", "status": "pending" }
          ],
          "reviewers": ["morgan"],
          "comments": [],
          "author": "taylor"
        }
      ],
      "issues": [
        {
          "number": 130,
          "title": "Regression: modal steals focus from address bar (Safari)",
          "state": "open",
          "labels": ["bug", "a11y"],
          "comments": []
        }
      ],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "web-ui",
          "messages": [
            {
              "id": "m1",
              "author": "taylor",
              "text": "a11y check still pending on #201, train boards Thursday 12:00 UTC",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [
        {
          "id": "e1",
          "title": "Release train boards",
          "start": "2026-11-19T12:00:00+00:00",
          "end": "2026-11-19T13:00:00+00:00",
          "attendees": [],
          "status": "confirmed"
        }
      ],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-17T11:00:00Z",
      "app": "repo",
      "function": "get_checks",
      "parameter": { "number": 201 }
    },
    {
      "event_id": "e2",
      "time": "2026-11-17T11:03:00Z",
      "app": "chat",
      "function": "search_messages",
      "parameter": { "query": "train" }
    }
  ],
  "notes": "Risk: a11y pending + known Safari regression; shipping without green a11y likely regresses #130."
}
```

`scenarios/platform-cve-merge-01.json`:

```json
{
  "id": "platform-cve-merge-01",
  "workspace": "platform",
  "need_type": "opportunity",
  "persona": { "name": "Riley Chen", "role": "platform engineer", "team": "build-fleet" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 77,
          "title": "bump libxml2 to 2.13.4 (CVE-2026-1234)",
          "branch": "deps/libxml2-2.13.4",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [
            { "name": "ci", "status": "pass" },
            { "name": "security", "status": "pass" }
          ],
          "reviewers": ["riley"],
          "comments": [],
          "author": "dependabot"
        }
      ],
      "issues": [
        {
          "number": 401,
          "title": "CVE-2026-1234 tracked for libxml2",
          "state": "open",
          "labels": ["security", "cve"],
          "comments": []
        }
      ],
      "commits": [
        { "sha": "feedbee", "message": "wip: local libxml experiments", "reverted": false }
      ]
    },
    "chat": {
      "channels": [
        {
          "name": "build-fleet",
          "messages": [
            {
              "id": "m1",
              "author": "security-bot",
              "text": "CVE-2026-1234 fix PR #77 is green — merge window open until Friday",
              "pinned": true
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-11T16:00:00Z",
      "app": "repo",
      "function": "get_pr",
      "parameter": { "number": 77 }
    },
    {
      "event_id": "e2",
      "time": "2026-11-11T16:01:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "build-fleet", "message_id": "m1" }
    }
  ],
  "notes": "Opportunity: green security PR during open window — merge (L3) vs over-waiting."
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scenario_corpus.py -v` → PASS
Run: `proactive-bench validate` (or `python -m proactive_bench.cli validate`) → exit 0, `5 scenarios, 0 error(s)`
Run: `ruff check src tests` → clean

- [ ] **Step 5: Commit**

```bash
git add scenarios/backend-ci-freeze-01.json scenarios/sre-latency-page-01.json scenarios/lead-review-queue-01.json scenarios/frontend-release-train-01.json scenarios/platform-cve-merge-01.json tests/test_scenario_corpus.py
git commit -m "Phase 3: initial workspace scenarios"
```

---

### Task 5: five more scenarios + Phase 3 exit floors

**Files:**
- Create: `scenarios/backend-dep-stale-01.json`
- Create: `scenarios/sre-handoff-risk-01.json`
- Create: `scenarios/lead-standup-opp-01.json`
- Create: `scenarios/frontend-design-block-01.json`
- Create: `scenarios/platform-fleet-block-01.json`
- Modify: `tests/test_scenario_corpus.py` (floors → ≥10, all need types, ≥2 per workspace)
- Modify: `README.md` (badge line 8, roadmap rows 99–101)
- Modify: `src/proactive_bench/cli.py:37` phase string → `Phase 3`
- Modify: `scenarios/README.md` (layout note → validated)

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: ≥10 scenarios; every need type present; README/CLI status at Phase 3.

**Coverage plan (10 total):**

| # | id | workspace | need_type |
|---|----|-----------|-----------|
| 1 | backend-ci-freeze-01 | backend | blockage |
| 2 | sre-latency-page-01 | sre | blockage |
| 3 | lead-review-queue-01 | lead | dependency |
| 4 | frontend-release-train-01 | frontend | risk |
| 5 | platform-cve-merge-01 | platform | opportunity |
| 6 | backend-dep-stale-01 | backend | dependency |
| 7 | sre-handoff-risk-01 | sre | risk |
| 8 | lead-standup-opp-01 | lead | opportunity |
| 9 | frontend-design-block-01 | frontend | blockage |
| 10 | platform-fleet-block-01 | platform | dependency |

Totals: dependency 3 · blockage 3 · risk 2 · opportunity 2; 2 per workspace.

- [ ] **Step 1: Tighten the corpus test (fails until 10 exist)**

Replace/extend `tests/test_scenario_corpus.py` helpers and floors:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenario_corpus.py -v`
Expected: FAIL — `scenario_count` is 5 (`assert 5 >= 10`)

- [ ] **Step 3: Write the five additional scenario JSON files**

`scenarios/backend-dep-stale-01.json`:

```json
{
  "id": "backend-dep-stale-01",
  "workspace": "backend",
  "need_type": "dependency",
  "persona": { "name": "Sam Okonkwo", "role": "backend engineer", "team": "payments" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 150,
          "title": "use new fraud-sdk batch API",
          "branch": "feat-fraud-batch",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [{ "name": "ci", "status": "pass" }],
          "reviewers": ["eda"],
          "comments": [],
          "author": "sam"
        }
      ],
      "issues": [],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "payments-eng",
          "messages": [
            {
              "id": "m1",
              "author": "eda",
              "text": "fraud-sdk batch API docs PR is still open on their side, ETA unknown",
              "pinned": false
            },
            {
              "id": "m2",
              "author": "sam",
              "text": "so #150 stays blocked until they land docs?",
              "pinned": false,
              "thread_id": "m1"
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": { "events": [], "focus_blocks": [] }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-14T13:00:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "payments-eng", "message_id": "m1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-14T13:02:00Z",
      "app": "repo",
      "function": "get_pr",
      "parameter": { "number": 150 }
    }
  ],
  "notes": "Dependency: external team docs gate the PR; comment/ask vs silent wait."
}
```

`scenarios/sre-handoff-risk-01.json`:

```json
{
  "id": "sre-handoff-risk-01",
  "workspace": "sre",
  "need_type": "risk",
  "persona": { "name": "Alex Rivera", "role": "site reliability engineer", "team": "edge" },
  "apps": {
    "repo": { "prs": [], "issues": [], "commits": [] },
    "chat": {
      "channels": [
        {
          "name": "sre-oncall",
          "messages": [
            {
              "id": "m1",
              "author": "casey",
              "text": "handing off in 30m — edge-gateway still firing a1, I have not acked yet",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [
        {
          "id": "a1",
          "severity": "sev3",
          "service": "edge-gateway",
          "status": "firing",
          "summary": "elevated 5xx during handoff"
        }
      ],
      "runbooks": [
        {
          "id": "rb-handoff",
          "title": "On-call handoff checklist",
          "steps": ["ack open alerts", "page_log notes", "freeze-window check"]
        }
      ],
      "deploy_windows": [
        {
          "id": "dw1",
          "start": "2026-11-15T20:00:00Z",
          "end": "2026-11-15T22:00:00Z",
          "frozen": true
        }
      ],
      "rollback_plans": [],
      "oncall": { "primary": "casey", "paged": false, "page_log": [] }
    },
    "cal": { "events": [], "focus_blocks": [] }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-15T19:30:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "sre-oncall", "message_id": "m1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-15T19:31:00Z",
      "app": "pager",
      "function": "get_alert",
      "parameter": { "id": "a1" }
    }
  ],
  "notes": "Risk: unacked sev3 at handoff during frozen deploy window; ack/page vs assume next shift sees it."
}
```

`scenarios/lead-standup-opp-01.json`:

```json
{
  "id": "lead-standup-opp-01",
  "workspace": "lead",
  "need_type": "opportunity",
  "persona": { "name": "Jordan Lee", "role": "tech lead", "team": "platform-experience" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 58,
          "title": "flaky canary metric filter",
          "branch": "fix-canary-filter",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [{ "name": "ci", "status": "pass" }],
          "reviewers": [],
          "comments": [],
          "author": "noor"
        }
      ],
      "issues": [],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "team-lead",
          "messages": [
            {
              "id": "m1",
              "author": "noor",
              "text": "#58 is green and unreviewed — can we get eyes before standup?",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [
        {
          "id": "e1",
          "title": "Team standup",
          "start": "2026-11-17T09:30:00+00:00",
          "end": "2026-11-17T09:45:00+00:00",
          "attendees": ["jordan", "noor", "kai"],
          "status": "confirmed"
        }
      ],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-17T09:00:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "team-lead", "message_id": "m1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-17T09:01:00Z",
      "app": "repo",
      "function": "get_pr",
      "parameter": { "number": 58 }
    }
  ],
  "notes": "Opportunity: green unreviewed PR + standup in 30m — request_review now vs discuss later."
}
```

`scenarios/frontend-design-block-01.json`:

```json
{
  "id": "frontend-design-block-01",
  "workspace": "frontend",
  "need_type": "blockage",
  "persona": { "name": "Morgan Blake", "role": "frontend engineer", "team": "web-ui" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 210,
          "title": "redesign settings layout",
          "branch": "feat-settings-redesign",
          "base": "main",
          "draft": true,
          "status": "open",
          "checks": [],
          "reviewers": [],
          "comments": [],
          "author": "morgan"
        }
      ],
      "issues": [
        {
          "number": 140,
          "title": "Settings redesign blocked on Figma spec v3",
          "state": "open",
          "labels": ["design"],
          "comments": []
        }
      ],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "web-ui",
          "messages": [
            {
              "id": "m1",
              "author": "casey-design",
              "text": "Figma v3 for settings is still in review with legal, no ETA",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": {
      "events": [
        {
          "id": "e1",
          "title": "Design review",
          "start": "2026-11-18T14:00:00+00:00",
          "end": "2026-11-18T15:00:00+00:00",
          "attendees": ["morgan", "casey-design"],
          "status": "confirmed"
        }
      ],
      "focus_blocks": []
    }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-17T15:00:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "web-ui", "message_id": "m1" }
    },
    {
      "event_id": "e2",
      "time": "2026-11-17T15:01:00Z",
      "app": "repo",
      "function": "get_issue",
      "parameter": { "number": 140 }
    }
  ],
  "notes": "Blockage: design/legal gate on draft PR; notify or propose reschedule vs silent stall."
}
```

`scenarios/platform-fleet-block-01.json`:

```json
{
  "id": "platform-fleet-block-01",
  "workspace": "platform",
  "need_type": "dependency",
  "persona": { "name": "Riley Chen", "role": "platform engineer", "team": "build-fleet" },
  "apps": {
    "repo": {
      "prs": [
        {
          "number": 80,
          "title": "roll out runner image 2026.11",
          "branch": "chore-runner-2026-11",
          "base": "main",
          "draft": false,
          "status": "open",
          "checks": [
            { "name": "ci", "status": "pending" },
            { "name": "fleet-canary", "status": "fail" }
          ],
          "reviewers": ["riley"],
          "comments": [],
          "author": "ci-bot"
        }
      ],
      "issues": [
        {
          "number": 410,
          "title": "Canary runners OOM on arm64 large",
          "state": "open",
          "labels": ["infra"],
          "comments": []
        }
      ],
      "commits": []
    },
    "chat": {
      "channels": [
        {
          "name": "build-fleet",
          "messages": [
            {
              "id": "m1",
              "author": "ci-bot",
              "text": "fleet-canary failed on #80 — waiting on arm64 memory fix from cloud team",
              "pinned": false
            }
          ]
        }
      ]
    },
    "pager": {
      "alerts": [],
      "runbooks": [],
      "deploy_windows": [],
      "rollback_plans": [],
      "oncall": { "primary": null, "paged": false, "page_log": [] }
    },
    "cal": { "events": [], "focus_blocks": [] }
  },
  "events": [
    {
      "event_id": "e1",
      "time": "2026-11-16T08:00:00Z",
      "app": "repo",
      "function": "get_checks",
      "parameter": { "number": 80 }
    },
    {
      "event_id": "e2",
      "time": "2026-11-16T08:02:00Z",
      "app": "chat",
      "function": "get_thread",
      "parameter": { "channel": "build-fleet", "message_id": "m1" }
    }
  ],
  "notes": "Dependency: external cloud-team fix gates canary; comment/track vs force-merge red canary."
}
```

- [ ] **Step 4: Bump Phase 3 status (README + CLI + scenarios README)**

`README.md` line 8:

```markdown
![Status: Phase 3](https://img.shields.io/badge/status-phase%203-yellow)
```

`README.md` roadmap rows:

```markdown
| 2 | Simulated apps + parser + L0–L3 | ✅ |
| 3 | Scenarios (5 workspaces → ~30) | ✅ **current** |
```

`src/proactive_bench/cli.py` phase string:

```python
        f"(see SPEC.md phases; currently Phase 3).",
```

`scenarios/README.md` — replace body with:

```markdown
# scenarios/

Self-contained scenario JSON files (SPEC §3.5). Validated by
`proactive-bench validate` (schema, allowlisted events, unique ids).

Layout:

```
scenarios/
├── README.md
└── <scenario-id>.json    # id must match filename stem
```

Workspaces: backend · sre · lead · frontend · platform
Need types: dependency · blockage · risk · opportunity

Phase 3 ships ≥10 scenarios; grow toward ~30 with the same schema.
```

- [ ] **Step 5: Run full verification**

Run: `pytest -v` → all pass
Run: `ruff check src tests` → clean
Run: `python -m proactive_bench.cli validate` → exit 0, `10 scenarios, 0 error(s)`

- [ ] **Step 6: Commit**

```bash
git add scenarios/backend-dep-stale-01.json scenarios/sre-handoff-risk-01.json scenarios/lead-standup-opp-01.json scenarios/frontend-design-block-01.json scenarios/platform-fleet-block-01.json tests/test_scenario_corpus.py README.md src/proactive_bench/cli.py scenarios/README.md
git commit -m "Phase 3: corpus floors, status bump"
```

---

## Self-review notes

- **Spec coverage:** §3.2 five workspaces (Tasks 4–5 fixtures + corpus tests), §3.3 four need types (coverage test), §3.5 schema (Task 1 exact keys + events via `parse_command`), §9 scenario test layer (`test_scenario_corpus` offline over `scenarios/**`), §11 Phase 3 exit `cli validate` passes (Task 3 + verification), ≥10 scenarios (Task 5 floor). Checkpoints/prefix semantics are Phase 4 runner concerns — out of scope here beyond documenting increasing event times as an authoring rule.
- **No placeholders:** every task has full test and implementation code; all ten scenario JSONs are complete.
- **Type consistency:** `ScenarioError`, `validate_scenario`, `validate_directory`, `DirectoryReport` used identically across Tasks 1–3; CLI consumes only Task 2/3 exports; corpus tests call `validate_directory` only.
- **Phase 2 reuse:** `parse_command` + `INITIAL_STATES` are the allowlist/shape authorities — no duplicated tool tables.
- **YAGNI:** no JSON-Schema library, no recursive workspace subdirs, no balance checker in the validator (floors live in corpus tests).
