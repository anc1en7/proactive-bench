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
    except (OSError, UnicodeDecodeError) as exc:
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
