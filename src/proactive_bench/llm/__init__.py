"""OpenAI-compatible LLM client (OmniRoute / any OpenAI-compatible endpoint)."""

from proactive_bench.llm.client import (
    DEFAULT_BACKOFF_BASE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_MODEL,
    ENV_TIMEOUT,
    LLMConfigError,
    LLMSettings,
    OpenAICompatClient,
    RateLimitError,
    settings_from_env,
)
from proactive_bench.llm.json_utils import (
    ChatClient,
    JsonChatResult,
    JSONExtractError,
    chat_json,
    extract_json_text,
    parse_json_text,
)

__all__ = [
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT",
    "ENV_API_KEY",
    "ENV_BASE_URL",
    "ENV_MODEL",
    "ENV_TIMEOUT",
    "ChatClient",
    "JSONExtractError",
    "LLMConfigError",
    "LLMSettings",
    "JsonChatResult",
    "OpenAICompatClient",
    "RateLimitError",
    "chat_json",
    "extract_json_text",
    "parse_json_text",
    "settings_from_env",
]
