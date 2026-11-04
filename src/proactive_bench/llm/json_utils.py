"""JSON extraction and parsing for agent/judge model replies."""

from __future__ import annotations

import json
import re
from typing import Any

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
