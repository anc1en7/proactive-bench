"""Simulated team-chat app: channels, messages, threads, pins."""

from __future__ import annotations

from typing import Any

from proactive_bench.env.base import (
    InvalidCommandError,
    ToolExecutionError,
    ToolSpec,
    next_id,
)


def initial_state() -> dict[str, Any]:
    return {"channels": []}


def _find_channel(state: dict[str, Any], name: Any) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise InvalidCommandError(f"channel must be a non-empty string, got {name!r}")
    for channel in state.setdefault("channels", []):
        if channel.get("name") == name:
            return channel
    known = ", ".join(str(c.get("name")) for c in state.get("channels", [])) or "none"
    raise ToolExecutionError(f"unknown channel {name!r} (known: {known})")


def _find_message(channel: dict[str, Any], message_id: Any) -> dict[str, Any]:
    if not isinstance(message_id, str) or not message_id:
        raise InvalidCommandError(f"message_id must be a non-empty string, got {message_id!r}")
    for message in channel.setdefault("messages", []):
        if message.get("id") == message_id:
            return message
    raise ToolExecutionError(f"unknown message {message_id!r} in channel {channel.get('name')!r}")


def _require_text(params: dict[str, Any]) -> str:
    text = params["text"]
    if not isinstance(text, str) or not text:
        raise InvalidCommandError(f"text must be a non-empty string, got {text!r}")
    return text


# --- reads -----------------------------------------------------------------


def _list_channels(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channels = [
        {"name": c.get("name", ""), "message_count": len(c.get("messages", []))}
        for c in state.setdefault("channels", [])
    ]
    return {"channels": channels}


def _get_thread(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    root = _find_message(channel, params["message_id"])
    root_id = root.get("thread_id", root.get("id"))
    messages = [
        m
        for m in channel.get("messages", [])
        if m.get("thread_id", m.get("id")) == root_id
    ]
    return {"messages": messages}


def _search_messages(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    query = params["query"]
    if not isinstance(query, str) or not query:
        raise InvalidCommandError(f"query must be a non-empty string, got {query!r}")
    needle = query.lower()
    matches = []
    for channel in state.setdefault("channels", []):
        for message in channel.get("messages", []):
            if needle in str(message.get("text", "")).lower():
                matches.append({"channel": channel.get("name"), "message": message})
    return {"matches": matches}


# --- writes ----------------------------------------------------------------


def _send_message(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    text = _require_text(params)
    messages = channel.setdefault("messages", [])
    message = {
        "id": next_id(messages, "m"),
        "author": params.get("author", "agent"),
        "text": text,
        "pinned": False,
    }
    messages.append(message)
    return {"ok": True, "message": message}


def _post_standup_update(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    result = _send_message(state, params)
    message = result["message"]
    message["kind"] = "standup"
    return result


def _pin_message(state: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    channel = _find_channel(state, params["channel"])
    message = _find_message(channel, params["message_id"])
    message["pinned"] = True
    return {"ok": True, "message": message}


TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(app="chat", name="list_channels", kind="read", handler=_list_channels),
    ToolSpec(
        app="chat",
        name="get_thread",
        kind="read",
        handler=_get_thread,
        required=("channel", "message_id"),
    ),
    ToolSpec(
        app="chat",
        name="search_messages",
        kind="read",
        handler=_search_messages,
        required=("query",),
    ),
    ToolSpec(
        app="chat",
        name="send_message",
        kind="write",
        handler=_send_message,
        required=("channel", "text"),
    ),
    ToolSpec(
        app="chat",
        name="post_standup_update",
        kind="write",
        handler=_post_standup_update,
        required=("channel", "text"),
    ),
    ToolSpec(
        app="chat",
        name="pin_message",
        kind="write",
        handler=_pin_message,
        required=("channel", "message_id"),
    ),
)
