"""Tests for JSON extraction/parsing helpers (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.llm.json_utils import (
    JSONExtractError,
    chat_json,
    extract_json_text,
    parse_json_text,
)


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


# --- Task 5: chat_json bounded-retry helper -------------------------------


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append(list(messages))
        return self.replies.pop(0)


def test_chat_json_success_first_attempt() -> None:
    client = FakeClient(['{"a": 1}'])
    messages = [{"role": "user", "content": "json please"}]
    result = chat_json(client, messages)
    assert result.ok is True
    assert result.data == {"a": 1}
    assert result.attempts == 1
    assert result.error is None
    assert len(client.calls) == 1
    assert client.calls[0] == messages
    assert messages == [{"role": "user", "content": "json please"}]  # not mutated


def test_chat_json_accepts_fenced_reply() -> None:
    client = FakeClient(['```json\n{"a": 1}\n```'])
    result = chat_json(client, [{"role": "user", "content": "hi"}])
    assert result.ok is True
    assert result.data == {"a": 1}


def test_chat_json_retries_with_nudge_message() -> None:
    client = FakeClient(["not json", '{"a": 2}'])
    messages = [{"role": "user", "content": "hi"}]
    result = chat_json(client, messages)
    assert result.ok is True
    assert result.data == {"a": 2}
    assert result.attempts == 2
    assert len(client.calls[1]) == len(messages) + 2
    assert client.calls[1][-1]["role"] == "user"
    assert "JSON" in client.calls[1][-1]["content"]


def test_chat_json_exhaustion_returns_not_ok() -> None:
    client = FakeClient(["x", "y", "z"])
    result = chat_json(client, [{"role": "user", "content": "hi"}])
    assert result.ok is False
    assert result.data is None
    assert result.attempts == 3
    assert result.raw == "z"
    assert result.error is not None
    assert "no valid JSON" in result.error


def test_chat_json_validate_rejects_then_accepts() -> None:
    client = FakeClient(['{"a": 1}', '{"a": 2}'])
    result = chat_json(
        client,
        [{"role": "user", "content": "hi"}],
        validate=lambda data: data.get("a") == 2,
    )
    assert result.ok is True
    assert result.data == {"a": 2}
    assert result.attempts == 2


def test_chat_json_validate_always_rejects() -> None:
    client = FakeClient(['{"a": 1}', '{"a": 1}', '{"a": 1}'])
    result = chat_json(
        client,
        [{"role": "user", "content": "hi"}],
        validate=lambda data: False,
    )
    assert result.ok is False
    assert result.error is not None
    assert "validation failed" in result.error


def test_chat_json_rejects_nonpositive_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        chat_json(FakeClient([]), [{"role": "user", "content": "hi"}], max_attempts=0)
