"""Tests for the simulated chat app (offline)."""

from __future__ import annotations

import pytest

from proactive_bench.env import chat
from proactive_bench.env.base import ToolExecutionError, ToolSpec


def get(name: str) -> ToolSpec:
    for spec in chat.TOOLS:
        if spec.name == name:
            return spec
    raise AssertionError(f"missing tool {name}")


def seed_state() -> dict:
    return {
        "channels": [
            {
                "name": "eng",
                "messages": [
                    {"id": "m1", "author": "alice", "text": "CI is red", "pinned": False},
                    {
                        "id": "m2",
                        "author": "bob",
                        "text": "looking",
                        "pinned": False,
                        "thread_id": "m1",
                    },
                    {
                        "id": "m3",
                        "author": "carol",
                        "text": "same here",
                        "pinned": False,
                        "thread_id": "m1",
                    },
                    {"id": "m5", "author": "dan", "text": "lunch?", "pinned": False},
                ],
            },
            {"name": "random", "messages": []},
        ]
    }


def test_tools_match_expected_names_and_kinds() -> None:
    reads = {s.name for s in chat.TOOLS if s.kind == "read"}
    writes = {s.name for s in chat.TOOLS if s.kind == "write"}
    assert reads == {"list_channels", "get_thread", "search_messages"}
    assert writes == {"send_message", "post_standup_update", "pin_message"}


def test_initial_state_is_fresh_each_call() -> None:
    a, b = chat.initial_state(), chat.initial_state()
    assert a == {"channels": []}
    a["channels"].append("x")
    assert b["channels"] == []


def test_list_channels_reports_counts() -> None:
    result = get("list_channels").handler(seed_state(), {})
    assert result["channels"] == [
        {"name": "eng", "message_count": 4},
        {"name": "random", "message_count": 0},
    ]


def test_get_thread_returns_root_and_replies() -> None:
    state = seed_state()
    result = get("get_thread").handler(state, {"channel": "eng", "message_id": "m1"})
    assert [m["id"] for m in result["messages"]] == ["m1", "m2", "m3"]
    from_reply = get("get_thread").handler(state, {"channel": "eng", "message_id": "m2"})
    assert [m["id"] for m in from_reply["messages"]] == ["m1", "m2", "m3"]
    alone = get("get_thread").handler(state, {"channel": "eng", "message_id": "m5"})
    assert [m["id"] for m in alone["messages"]] == ["m5"]


def test_get_thread_unknown_channel_or_message() -> None:
    state = seed_state()
    with pytest.raises(ToolExecutionError, match="channel"):
        get("get_thread").handler(state, {"channel": "nope", "message_id": "m1"})
    with pytest.raises(ToolExecutionError, match="message"):
        get("get_thread").handler(state, {"channel": "eng", "message_id": "m99"})


def test_search_messages_case_insensitive() -> None:
    result = get("search_messages").handler(seed_state(), {"query": "CI"})
    assert [m["message"]["id"] for m in result["matches"]] == ["m1"]
    assert get("search_messages").handler(seed_state(), {"query": "zzz"})["matches"] == []


def test_send_message_appends_with_next_id() -> None:
    state = seed_state()
    result = get("send_message").handler(state, {"channel": "eng", "text": "on it"})
    assert result["ok"] is True
    assert result["message"]["id"] == "m6"
    assert result["message"]["author"] == "agent"
    assert state["channels"][0]["messages"][-1]["text"] == "on it"
    with pytest.raises(ToolExecutionError, match="channel"):
        get("send_message").handler(state, {"channel": "nope", "text": "x"})


def test_post_standup_update_marks_kind() -> None:
    state = seed_state()
    result = get("post_standup_update").handler(
        state, {"channel": "eng", "text": "yesterday: shipped"}
    )
    assert result["message"]["kind"] == "standup"
    assert result["message"]["author"] == "agent"


def test_pin_message_sets_pinned() -> None:
    state = seed_state()
    get("pin_message").handler(state, {"channel": "eng", "message_id": "m5"})
    assert state["channels"][0]["messages"][3]["pinned"] is True
    with pytest.raises(ToolExecutionError):
        get("pin_message").handler(state, {"channel": "eng", "message_id": "m99"})


def test_send_message_rejects_empty_text() -> None:
    from proactive_bench.env.base import InvalidCommandError

    with pytest.raises(InvalidCommandError):
        get("send_message").handler(seed_state(), {"channel": "eng", "text": ""})
