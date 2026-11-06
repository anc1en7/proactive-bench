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
