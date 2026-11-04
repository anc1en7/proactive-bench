"""Tests for JSON extraction/parsing helpers (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.llm.json_utils import JSONExtractError, extract_json_text, parse_json_text


def test_parse_plain_object() -> None:
    assert parse_json_text('{"a": 1}') == {"a": 1}


def test_parse_fenced_block() -> None:
    text = 'Here you go:\n```json\n{"a": 1}\n```\nDone.'
    assert parse_json_text(text) == {"a": 1}


def test_parse_object_wrapped_in_prose() -> None:
    text = 'The answer is {"a": [1, 2]} per the log above.'
    assert parse_json_text(text) == {"a": [1, 2]}


def test_parse_array_root() -> None:
    assert parse_json_text("result: [1, 2, 3]") == [1, 2, 3]


def test_braces_inside_strings_do_not_confuse_extractor() -> None:
    text = 'note: {"s": "close } and open { brace"} end'
    assert parse_json_text(text) == {"s": "close } and open { brace"}


def test_no_json_raises() -> None:
    with pytest.raises(JSONExtractError):
        extract_json_text("no json here at all")


def test_malformed_json_raises() -> None:
    with pytest.raises(JSONExtractError):
        extract_json_text('{"a": ')


def test_invalid_fence_falls_through_to_object() -> None:
    text = '```json\nnot json\n```\nreal: {"a": 1}'
    assert parse_json_text(text) == {"a": 1}
