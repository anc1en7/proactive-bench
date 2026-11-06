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
