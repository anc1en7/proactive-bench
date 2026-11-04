"""Tests for the OpenAI-compatible LLM client (offline, mocked transport)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from proactive_bench.llm.client import (
    LLMConfigError,
    LLMSettings,
    OpenAICompatClient,
    RateLimitError,
    settings_from_env,
)

_ENV_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT")


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neutralize any real .env so tests are deterministic on dev machines.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> OpenAICompatClient:
    settings = LLMSettings(base_url="https://llm.test/v1", api_key="sk-test", model="m1")
    kwargs.setdefault("sleep", lambda _seconds: None)
    return OpenAICompatClient(settings, transport=httpx.MockTransport(handler), **kwargs)


def ok_reply(content: str = "ok") -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def always(status: int, **kwargs: Any) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, **kwargs)

    return handler


# --- Task 1: settings & env loading ---------------------------------------


def test_settings_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    with pytest.raises(LLMConfigError) as exc:
        settings_from_env()
    assert "LLM_BASE_URL" in str(exc.value)


def test_settings_rejects_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "changeme")
    monkeypatch.setenv("LLM_MODEL", "rag")
    with pytest.raises(LLMConfigError):
        settings_from_env()


def test_settings_from_env_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.internal/v1/")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_MODEL", "rag")
    settings = settings_from_env()
    assert settings.base_url == "https://llm.internal/v1"
    assert settings.model == "rag"
    assert settings.timeout == 120.0
    assert settings.host == "llm.internal"
    assert "sk-secret" not in repr(settings)


def test_settings_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.internal/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    monkeypatch.setenv("LLM_MODEL", "rag")
    monkeypatch.setenv("LLM_TIMEOUT", "30")
    assert settings_from_env().timeout == 30.0
    monkeypatch.setenv("LLM_TIMEOUT", "not-a-number")
    with pytest.raises(LLMConfigError):
        settings_from_env()


# --- Task 2: chat client core ---------------------------------------------


def test_chat_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return ok_reply("hello")

    client = make_client(handler)
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"


def test_chat_sends_model_auth_and_params() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers["authorization"]
        captured["body"] = json.loads(request.content)
        return ok_reply()

    client = make_client(handler)
    client.chat(
        [{"role": "user", "content": "hi"}],
        model="override-model",
        temperature=0.0,
        max_tokens=64,
    )
    body = captured["body"]
    assert captured["auth"] == "Bearer sk-test"
    assert body["model"] == "override-model"
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 64


def test_models_lists_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "rag"}, {"id": "aug/haiku4.5"}]})

    client = make_client(handler)
    assert client.models() == ["rag", "aug/haiku4.5"]


def test_context_manager() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_reply()

    with make_client(handler) as client:
        assert client.chat([{"role": "user", "content": "hi"}]) == "ok"


# --- Task 3: retry policy --------------------------------------------------


def test_chat_retries_429_then_success() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return ok_reply("finally")

    client = make_client(handler, sleep=sleeps.append)
    assert client.chat([{"role": "user", "content": "hi"}]) == "finally"
    assert len(attempts) == 3
    assert len(sleeps) == 2
    assert all(delay >= 0 for delay in sleeps)


def test_chat_retries_5xx_with_exponential_backoff() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503)
        return ok_reply()

    client = make_client(handler, sleep=sleeps.append, backoff_base=1.0)
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert sleeps == [1.0]


def test_chat_rate_limit_exhaustion_raises() -> None:
    sleeps: list[float] = []
    client = make_client(always(429, headers={"Retry-After": "7"}), sleep=sleeps.append)
    with pytest.raises(RateLimitError) as exc:
        client.chat([{"role": "user", "content": "hi"}])
    assert exc.value.retry_after == 7.0
    assert len(sleeps) == 4  # DEFAULT_MAX_RETRIES


def test_chat_4xx_does_not_retry() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": "bad request"})

    client = make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(attempts) == 1


def test_chat_respects_max_retries_override() -> None:
    sleeps: list[float] = []
    client = make_client(always(500), sleep=sleeps.append, max_retries=1, backoff_base=1.0)
    with pytest.raises(httpx.HTTPStatusError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(sleeps) == 1
