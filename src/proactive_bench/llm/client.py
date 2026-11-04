"""OpenAI-compatible LLM client (OmniRoute-friendly) with retries."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

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
