"""OpenAI-compatible LLM client (OmniRoute-friendly) with retries."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"
ENV_TIMEOUT = "LLM_TIMEOUT"
REQUIRED_ENV = (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)

DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_BACKOFF_BASE = 1.0
_MAX_DELAY = 60.0


class LLMConfigError(RuntimeError):
    """Raised when the LLM endpoint is not configured."""


class RateLimitError(RuntimeError):
    """429 after retries were exhausted; ``retry_after`` is best-effort seconds."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _parse_timeout(raw: str | None) -> float:
    if raw is None:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{ENV_TIMEOUT} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise LLMConfigError(f"{ENV_TIMEOUT} must be positive, got {raw!r}")
    return value


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout: float = DEFAULT_TIMEOUT

    @property
    def host(self) -> str:
        """Host only — never log or persist credentials or full URLs."""
        return httpx.URL(self.base_url).host or ""


def settings_from_env() -> LLMSettings:
    """Read endpoint configuration from the environment (.env supported)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    missing = [key for key in REQUIRED_ENV if not os.getenv(key)]
    if missing:
        raise LLMConfigError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your endpoint details."
        )
    base_url = os.environ[ENV_BASE_URL].rstrip("/")
    api_key = os.environ[ENV_API_KEY]
    model = os.environ[ENV_MODEL]
    if api_key == "changeme" or model == "changeme" or base_url.endswith("example.com/v1"):
        raise LLMConfigError(
            f"{ENV_BASE_URL} / {ENV_API_KEY} / {ENV_MODEL} still contain placeholder "
            "values; fill in your real endpoint in .env."
        )
    return LLMSettings(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=_parse_timeout(os.getenv(ENV_TIMEOUT)),
    )


class OpenAICompatClient:
    """Minimal chat-completions client for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=settings.timeout,
            transport=transport,
        )

    def models(self) -> list[str]:
        response = self._client.get("/models")
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return [str(item["id"]) for item in payload.get("data", [])]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        body = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return str(payload["choices"][0]["message"]["content"])

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
