"""JSON extraction and parsing for agent/judge model replies."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONExtractError(ValueError):
    """No JSON object/array could be located in the model reply."""


def _balanced_slice(text: str, start: int) -> str | None:
    opening = text[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_text(text: str) -> str:
    """Locate the first parseable JSON object/array in ``text``."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    for index, char in enumerate(text):
        if char in "{[":
            chunk = _balanced_slice(text, index)
            if chunk is None:
                continue
            try:
                json.loads(chunk)
                return chunk
            except json.JSONDecodeError:
                continue
    raise JSONExtractError(f"no valid JSON found in reply: {text[:120]!r}")


def parse_json_text(text: str) -> Any:
    """Extract then parse JSON; raises ``JSONExtractError`` when absent."""
    return json.loads(extract_json_text(text))


class ChatClient(Protocol):
    """Structural type for anything with ``OpenAICompatClient.chat``'s signature."""

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...


_NUDGE = (
    "Your previous reply could not be parsed as JSON. "
    "Reply again with only a single valid JSON value and no extra text."
)


@dataclass(frozen=True)
class JsonChatResult:
    ok: bool
    data: Any | None
    attempts: int
    raw: str
    error: str | None


def chat_json(
    client: ChatClient,
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_attempts: int = 3,
    validate: Callable[[Any], bool] | None = None,
) -> JsonChatResult:
    """Call ``client.chat`` until the reply parses (and validates), bounded."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    history = list(messages)
    last_raw = ""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        last_raw = client.chat(
            history, model=model, temperature=temperature, max_tokens=max_tokens
        )
        try:
            data = parse_json_text(last_raw)
        except (JSONExtractError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        else:
            if validate is None or validate(data):
                return JsonChatResult(
                    ok=True, data=data, attempts=attempt, raw=last_raw, error=None
                )
            last_error = f"validation failed for: {last_raw[:120]!r}"
        history = [
            *history,
            {"role": "assistant", "content": last_raw},
            {"role": "user", "content": _NUDGE},
        ]
    return JsonChatResult(
        ok=False, data=None, attempts=max_attempts, raw=last_raw, error=last_error
    )
