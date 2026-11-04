"""Tests for env base types: ToolSpec, errors, helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from proactive_bench.env.base import (
    InvalidCommandError,
    MissingParameterError,
    ToolError,
    ToolExecutionError,
    ToolKindError,
    ToolSpec,
    UnknownAppError,
    UnknownToolError,
    as_int,
    next_id,
)


def _noop_handler(state: dict, params: dict) -> dict:
    return {"ok": True}


def make_spec(**kwargs: object) -> ToolSpec:
    defaults: dict = {
        "app": "repo",
        "name": "get_pr",
        "kind": "read",
        "handler": _noop_handler,
        "required": ("number",),
    }
    defaults.update(kwargs)
    return ToolSpec(**defaults)


def test_errors_are_tool_error_subclasses() -> None:
    for cls in (
        InvalidCommandError,
        UnknownAppError,
        UnknownToolError,
        MissingParameterError,
        ToolKindError,
        ToolExecutionError,
    ):
        assert issubclass(cls, ToolError)
        assert issubclass(cls, Exception)


def test_check_params_ok_when_present() -> None:
    make_spec().check_params({"number": 1})


def test_check_params_lists_missing_keys() -> None:
    with pytest.raises(MissingParameterError) as exc:
        make_spec(required=("number", "body")).check_params({"number": 1})
    assert "body" in str(exc.value)
    assert "number" not in str(exc.value).split(":")[-1]


def test_tool_spec_is_immutable() -> None:
    spec = make_spec()
    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_next_id_increments_numeric_suffix() -> None:
    items = [{"id": "m1"}, {"id": "m2"}, {"id": "other"}]
    assert next_id(items, "m") == "m3"


def test_next_id_starts_at_one_when_empty() -> None:
    assert next_id([], "e") == "e1"


def test_next_id_skips_non_matching_prefix() -> None:
    items = [{"id": "e2"}, {"id": "f9"}]
    assert next_id(items, "e") == "e3"


def test_next_id_with_empty_prefix() -> None:
    items = [{"id": "1"}, {"id": "4"}, {"id": "x"}]
    assert next_id(items, "") == "5"


def test_as_int_accepts_int_and_digit_string() -> None:
    assert as_int(7, "number") == 7
    assert as_int("7", "number") == 7
    assert as_int("-7", "number") == -7
    assert as_int(" 7 ", "number") == 7


def test_as_int_rejects_garbage() -> None:
    for bad in (None, 1.5, True, "abc", [1], "--7", "²"):
        with pytest.raises(InvalidCommandError):
            as_int(bad, "number")
