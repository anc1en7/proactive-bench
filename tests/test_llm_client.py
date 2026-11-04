"""Tests for the OpenAI-compatible LLM client (offline, mocked transport)."""

from __future__ import annotations

import pytest

from proactive_bench.llm.client import LLMConfigError, settings_from_env

_ENV_KEYS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT")


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Neutralize any real .env so tests are deterministic on dev machines.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None, raising=False)
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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
